import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
from xml.dom import minidom

#### Rutas carpetas y datos

data_adress = r"C:\Users\Víctor\Documents\Estudios\Prácticas Biomech\Proyecto_Xsens\Archivos\DATOS_VICTOR\MARCHA_IDA_SENSORES_DESORDENADOS"
results_rout = r"C:\Users\Víctor\Documents\Estudios\Prácticas Biomech\Proyecto_Xsens\Resultados"

final_excel = os.path.join(results_rout, "Excel_Datos.xlsx")
graphs_folder = os.path.join(results_rout, "Gráficas_PCA")
sto_movement = os.path.join(results_rout, "march_movement.sto")
sto_calibration = os.path.join(results_rout, "static_calibration.sto")
xml_placer = os.path.join(results_rout, "configuration_placer.xml")

if not os.path.exists(graphs_folder): os.makedirs(graphs_folder)

SEGMENTS_MAP = {
    '00B4EE25': 'calcn_l_imu',   '00B4EE20': 'calcn_r_imu',
    '00B4EE23': 'tibia_l_imu', '00B4EDFD': 'tibia_r_imu',
    '00B4EE01': 'femur_l_imu', '00B4EDFB': 'femur_r_imu',
    '00B4EE0C': 'pelvis_imu'       
}

# ------------------------------------------------------------------
# Segmentación de la señal
# ------------------------------------------------------------------

def segment_data(df, sensor_name, idx_turn = None):
    """
    Bases de la segmentación:
    - Basada en aceleración triaxial, comparando con 9.81"
    - Sistema de coordenadas ENU
    """
    # Calculamos la frecuencia de muestreo de la señal en Hz
    if 'SampleTimeFine' in df.columns:
        diff_time = df['SampleTimeFine'].diff().dropna()
        if not diff_time.empty and diff_time.mean() > 0:
            fs = 1 / (diff_time.mean() / 1_000_000)
        else:    
            fs = 100.0 # Por si los datos están corruptos
    else:
        fs = 100.0 # Por si no existe la columna

    # Calculamos el módulo de la aceleración triaxial
    acc_cols = get_cols(df, 'Acc_')
    val = np.linalg.norm(df[acc_cols].values, axis = 1) 

    # Suavizado de la señal -> Evitar falsos positivos
    w_size = int(0.3 * fs)
    smooth = pd.Series(val).rolling(window = w_size, center = True).mean().fillna(9.81)

    # Definimos el umbral de movimiento -> Detectar el inicio (desviaciónes en g de > 0.4 m/s^2)
    indices_mov = np.where(np.abs(smooth - 9.81) > 0.4)[0]

    if len(indices_mov) == 0:
        print(f"No se detectó movimiento en el sensor {sensor_name}. Usando valores por defecto")
        return (0, int(2*fs)), (int(2*fs)+1, len(df)-1)

    onset = indices_mov[0] # Donde empieza el movimiento
    margin = 0.5 # Para evitar ruidos a principio y fin de los periodos
    
    # FASE ESTÁTICA 
    static_start = int(0.3 * fs)
    static_end = onset - int(margin * fs) # Damos un pequeño margen al principio
    
    # Por si hay algún problema con los márgenes anteriores
    if static_end <= static_start:
        static_range = (0, max(10, int(onset * 0.7)))
    else: 
        static_range = (int(static_start), int(static_end))

    ### FASE DINÁMICA
    motion_start = onset + int(margin*fs) # Damos un margen para evitar errores
    
    # El límite de la marcha se marca por el inicio del giro. Si no existe, por el final del archivo
    motion_limit = idx_turn if idx_turn is not None else len(df)
    motion_end = motion_limit - int(margin*fs)

    # Verificamos que el rango de marcha es válido 
    if motion_end <= motion_start: # Si es muy corta
        motion_range = (onset, motion_limit)
    else:
        motion_range = (int(motion_start), int(motion_end))

    # Gráficas para validación
    plt.figure(figsize=(10,4))
    plt.plot(val, color = 'silver', alpha = 0.5, label = 'Total Acc.(Mag)')
    plt.plot(smooth, color = 'black', alpha = 0.8, label = 'Smoothed')
    plt.axvspan(static_range[0], static_range[1], color = 'green', alpha=0.2, label = 'N-Pose (Static)')
    plt.axvspan(motion_range[0], motion_range[1], color = 'blue', alpha=0.2, label = 'March (Functional)')
    if idx_turn:
        plt.axvline(idx_turn, color = 'red', linestyle = '--', label = 'Turn')

    plt.title(f'Test Segmentation: {sensor_name} ({fs:.1f} Hz)')
    plt.ylabel('Acceleration (m/s^2)')
    plt.xlabel('Frames')
    plt.legend()
    plt.grid(alpha = 0.2)
    plt.savefig(os.path.join(graphs_folder, f'Seg_{sensor_name}.png'))
    plt.close()

    return static_range, motion_range

# ------------------------------------------------------------------
# Lógica Matemática
# ------------------------------------------------------------------

### ------------- Carga de datos

files = glob.glob(os.path.join(data_adress, "*.txt"))
dict_dfs = {}
for f in files:
    dev_id = f.split('_')[-1].replace('.txt','')
    if dev_id in SEGMENTS_MAP:
        name = SEGMENTS_MAP[dev_id]
        skip_n = 0
        with open(f, 'r') as file:
            for i, line in enumerate(file):
                if 'PacketCounter' in line: skip_n = i; break
        df = pd.read_csv(f, skiprows = skip_n, sep = ';', low_memory = False)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors = 'coerce')
        dict_dfs[name] = df

### ------------- Detección de giro (Referencia: PELVIS)

yaw_p = np.unwrap(np.deg2rad(dict_dfs['pelvis_imu']['Yaw'].values))
idx_turn_global = np.where(np.abs(yaw_p - np.mean(yaw_p[20:120])) > 1.2)[0]
idx_turn_global = idx_turn_global[0] if len(idx_turn_global) >0 else None

### ------------- Funciones algebráicas

def get_cols(df, pattern):
    return [c for c in df.columns if pattern in c][:3]

def get_vertical_local(q_static):
    """
    Encuentra el eje local del sensor que mejor apunta al techo (Z global en ENU).
    Es infalible incluso si el sensor está de lado, boca abajo o inclinado.
    """
    # Ejes candidatos locales
    candidates = [
        np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    ]
    
    best_axis = None
    max_dot = -np.inf
    
    for c in candidates:
        # Llevamos el eje local al mundo global para ver cuánto se alinea con 'Arriba'
        v_global = q_static.apply(c)
        dot = v_global[2] # Componente Z global (el 'arriba' de XSens)
        
        # Nos quedamos con el eje que tenga mayor valor absoluto (el más vertical)
        if abs(dot) > max_dot:
            max_dot = abs(dot)
            # Si el valor era negativo, el eje apunta hacia abajo, así que lo invertimos
            best_axis = c if dot > 0 else -c
            
    return best_axis / np.linalg.norm(best_axis)

def get_global_forward_from_yaw(df, t_mov):
    """
    Calcula el rumbo (vector de avance) en el mundo global usando la brújula de XSens.
    """
    yaw_rad = np.deg2rad(df.iloc[t_mov[0]:t_mov[1]]['Yaw'].dropna().values)
    yaw_mean = np.arctan2(np.mean(np.sin(yaw_rad)), np.mean(np.cos(yaw_rad)))
    # Vector de avance en el mundo global (X, Y, Z)
    v_fwd_global = np.array([np.cos(yaw_mean), np.sin(yaw_mean), 0.0])
    return v_fwd_global, yaw_mean

def get_ml_axis_from_static(q_static, yaw_global):
    """
    Sin distinción _r/_l. El eje lateral derecho global es el mismo
    para todos los segmentos: es una propiedad del espacio, no del segmento.
    """
    v_right_global = np.array([np.sin(yaw_global), -np.cos(yaw_global), 0.0])
    v_ml_local = q_static.inv().apply(v_right_global)
    v_ml_local /= np.linalg.norm(v_ml_local)
    return v_ml_local


def get_orthonormal_basis(v_up, v_ml_local, v_fwd_ref_global, q_static):
    """
    Construye la base y luego verifica que X apunte al frente.
    Si no, invierte Z (ML) y recalcula. Así el signo de ML queda
    determinado por el resultado en X, no por una suposición anatómica.
    """
    y_axis = v_up / np.linalg.norm(v_up)

    z_axis = v_ml_local - np.dot(v_ml_local, y_axis) * y_axis
    z_axis /= np.linalg.norm(z_axis)

    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # Comprobamos que X apunta al frente en coordenadas globales
    x_global = q_static.apply(x_axis)
    if np.dot(x_global, v_fwd_ref_global) < 0:
        # X apunta atrás: invertimos Z y recalculamos X
        z_axis = -z_axis
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

    return R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))

### ------------- BUCLE DE CALIBRACIÓN

segment_ranges = {}
s2s_matrices = {} 
calib_log = []
q_cols_xsens = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']

# --- PASO 1: PELVIS (ANCLA EL RUMBO) ---
df_p = dict_dfs['pelvis_imu']
r_est_p, r_mov_p = segment_data(df_p, 'pelvis_imu', idx_turn_global)
segment_ranges['pelvis_imu'] = (r_est_p, r_mov_p)

q_static_p = R.from_quat(df_p.iloc[r_est_p[0]:r_est_p[1]][q_cols_xsens].dropna().values).mean()

# CAMBIO AQUÍ: Usamos la nueva función robusta
v_vert_p = get_vertical_local(q_static_p) 

v_fwd_global, yaw_maestro = get_global_forward_from_yaw(df_p, r_mov_p)

# Calibramos Pelvis (pasando el avance para validar su X)
v_ml_p = get_ml_axis_from_static(q_static_p, yaw_maestro)
s2s_matrices['pelvis_imu'] = get_orthonormal_basis(v_vert_p, v_ml_p, v_fwd_global, q_static_p)

# --- PASO 2: EL RESTO DE SENSORES ---
for seg, df in dict_dfs.items():
    if seg == 'pelvis_imu': continue
    
    r_est, r_mov = segment_data(df, seg, idx_turn_global)
    segment_ranges[seg] = (r_est, r_mov)

    q_static = R.from_quat(df.iloc[r_est[0]:r_est[1]][q_cols_xsens].dropna().values).mean()
    v_vert = get_vertical_local(q_static)

    # Obtenemos ML desde la brújula global
    v_ml = get_ml_axis_from_static(q_static, yaw_maestro)
    
    # Construcción validada: garantizamos que X apunte como la pelvis
    s2s_matrices[seg] = get_orthonormal_basis(v_vert, v_ml, v_fwd_global, q_static)

    # Log para el Excel
    mtx = s2s_matrices[seg].as_matrix()
    calib_log.append({
        'Segmento': seg,
        'V_UP_X': mtx[0,1], 'V_UP_Y': mtx[1,1], 'V_UP_Z': mtx[2,1],
        'V_ML_X': mtx[0,2], 'V_ML_Y': mtx[1,2], 'V_ML_Z': mtx[2,2]
    })

pd.DataFrame(calib_log).to_excel(final_excel, index=False)

# ------------------------------------------------------------------
# Exportación OpenSim
# ------------------------------------------------------------------

### ------------- Archivos .STO

def export_sto(dict_dfs, output_path, segment_ranges, s2s_matrices):
    min_len = min([len(df) for df in dict_dfs.values()])
    q_cols = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']
    
    data_rows = []
    is_static = "calibracion" in output_path or "static" in output_path
    frames_to_process = 1 if is_static else min_len

    for i in range(frames_to_process):
        row = [i/100.0]
        for seg, df in dict_dfs.items():
            if is_static:
                t = segment_ranges[seg][0]
                q_raw = R.from_quat(df.iloc[t[0]:t[1]][q_cols].dropna().values).mean()
            else:
                q_raw = R.from_quat(df.iloc[i][q_cols].values)

            # TRANSFORMACIÓN PURA: Dato * S2S
            # S2S ya contiene el Rumbo (Yaw) y la Verticalidad
            q_final_rot = q_raw * s2s_matrices[seg]
            q_f = q_final_rot.as_quat()

            if q_f[3] < 0: q_f = -q_f
            row.append(f"{q_f[3]},{q_f[0]},{q_f[1]},{q_f[2]}")

        data_rows.append(row)

    with open(output_path, 'w') as f:
        f.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.5\n")
        f.write(f"nRows={len(data_rows)}\nnColumns={len(dict_dfs)+1}\n")
        f.write("endheader\n")
        pd.DataFrame(data_rows, columns=['time'] + list(dict_dfs.keys())).to_csv(f, sep='\t', index=False)

# Llamadas finales
export_sto(dict_dfs, sto_movement, segment_ranges, s2s_matrices)
export_sto(dict_dfs, sto_calibration, segment_ranges, s2s_matrices)



### ------------- Archivo XML

def generate_xml_placer(xml_path, sto_calibration_name):
    root = ET.Element("OpenSimDocument", Version="40000")
    placer = ET.SubElement(root, "IMUPlacer")
    
    # El sensor que manda para orientar el modelo (Pelvis)
    ET.SubElement(placer, "base_imu_label").text = 'pelvis_imu'
    
    # El eje de avance del modelo (X en nuestro caso)
    ET.SubElement(placer, "base_heading_axis").text = 'x'
    
    # Rotación necesaria para pasar de Z-Up (IMU) a Y-Up (OpenSim)
    # -1.5707963267948966 rad = -90 grados en el eje X
    ET.SubElement(placer, "sensor_to_opensim_rotations").text = "-1.5707963267948966 0 0"
    
    # Nombre del archivo de calibración que acabamos de generar
    ET.SubElement(placer, "orientation_file_for_calibration").text = sto_calibration_name
    
    # (Opcional) Rango de tiempo para la calibración (usamos el primer frame)
    ET.SubElement(placer, "time_range").text = "0 0"

    # Escritura estética
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(xml_path, 'w') as f:
        f.write(xml_str)

# Llamada a la función
generate_xml_placer(xml_placer, os.path.basename(sto_calibration))

print(f"\n>>> PROCESO FINALIZADO CON ÉXITO")

# --------------

print("\n" + "="*50)
print("   DIAGNÓSTICO FINAL DE CALIBRACIÓN")
print("="*50)

# 1. TEST DE VERTICALIDAD GLOBAL (El modelo debe estar erguido)
print("\n--- TEST 1: VERTICALIDAD (Eje Y del hueso debe ser [0, 0, 1]) ---")
for seg, rot in s2s_matrices.items():
    t_st = segment_ranges[seg][0]
    q_st = R.from_quat(dict_dfs[seg].iloc[t_st[0]:t_st[1]][q_cols_xsens].dropna().values).mean()
    # En OpenSim, el eje Y es el que va hacia el techo. 
    # Comprobamos dónde apunta ese eje Y en el mundo real (ENU)
    up_world = (q_st * rot).apply([0, 1, 0])
    status = "OK" if up_world[2] > 0.9 else "REVISAR"
    print(f"{seg:15}: Up Global = {up_world.round(2)}  [{status}]")

# 2. TEST DE RUMBO GLOBAL (El modelo debe mirar al frente de la marcha)
print("\n--- TEST 2: RUMBO (Eje X del hueso debe ser el avance de la marcha) ---")
# El avance ideal es el que calculamos con la pelvis: v_fwd_global
for seg, rot in s2s_matrices.items():
    t_st = segment_ranges[seg][0]
    q_st = R.from_quat(dict_dfs[seg].iloc[t_st[0]:t_st[1]][q_cols_xsens].dropna().values).mean()
    # En OpenSim, el eje X es el avance.
    fwd_world = (q_st * rot).apply([1, 0, 0])
    
    # Comprobamos si el eje X del hueso se alinea con el avance maestro
    # El valor debe ser parecido a tu [0.82, 0.57, 0]
    error_avance = np.dot(fwd_world, v_fwd_global)
    status = "OK" if error_avance > 0.9 else "!!! SENTIDO INVERTIDO"
    print(f"{seg:15}: Frente Global = {fwd_world.round(2)}  [{status}]")

print("\n" + "="*50)
print("SI AMBOS TESTS DAN 'OK', EL MODELO CAMINARÁ PERFECTO")
print("="*50)
