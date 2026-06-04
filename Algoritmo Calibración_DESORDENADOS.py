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

data_adress = r"C:\Users\Víctor\Documents\Estudios\Prácticas Biomech\Proyecto_Xsens\Archivos\DATOS_VICTOR\MARCHA_IDA_SENSORES_ALINEADOS_SINRESET"
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

def get_cols(df, pattern):
    return [c for c in df.columns if pattern in c][:3] 

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

def get_vertical_local(q_static):
    """
    Extrae la dirección exacta del 'Techo' (Z global [0,0,1]) 
    desde la perspectiva local del sensor.
    """
    v_vert = q_static.inv().apply([0, 0, 1])
    return v_vert / np.linalg.norm(v_vert)

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
    Construye la base y garantiza que el eje X apunte al frente de la marcha.
    """
    # 1. Eje Y (UP)
    y_axis = v_up / np.linalg.norm(v_up)

    # 2. Eje Z (RIGHT) - Ortogonalizamos respecto a Y
    z_axis = v_ml_local - np.dot(v_ml_local, y_axis) * y_axis
    z_axis /= np.linalg.norm(z_axis)

    # 3. Eje X (FORWARD) -> X = Y x Z
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # --- SEGURO DE RUMBO ---
    # Si el X resultante apunta hacia atrás del avance de la pelvis, invertimos Z
    x_global = q_static.apply(x_axis)
    if np.dot(x_global, v_fwd_ref_global) < 0:
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

# AQUÍ SE DEFINE q_static_p
q_static_p = R.from_quat(df_p.iloc[r_est_p[0]:r_est_p[1]][q_cols_xsens].dropna().values).mean()

# Ahora usamos la función simplificada
v_vert_p = get_vertical_local(q_static_p)

# Obtenemos el rumbo global maestro (Brújula)
v_fwd_global, yaw_maestro = get_global_forward_from_yaw(df_p, r_mov_p)

# ML Pelvis
v_ml_p = get_ml_axis_from_static(q_static_p, yaw_maestro)

# Construcción de base S2S para la pelvis
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
    q_cols = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']
    is_static = "calibracion" in output_path or "static" in output_path
    min_len = min([len(df) for df in dict_dfs.values()])
    frames_to_process = 1 if is_static else min_len
    data_rows = []

    for i in range(frames_to_process):
        row = [i / 100.0]
        for seg, df in dict_dfs.items():
            if is_static:
                t = segment_ranges[seg][0]
                q_raw = R.from_quat(df.iloc[t[0]:t[1]][q_cols].dropna().values).mean()
            else:
                q_raw = R.from_quat(df.iloc[i][q_cols].values)

            # TRANSFORMACIÓN PURA: Solo el sensor por su matriz de calibración
            # No rotamos nada más aquí.
            q_final = q_raw * s2s_matrices[seg]
            q_f = q_final.as_quat()

            if q_f[3] < 0: q_f = -q_f
            row.append(f"{q_f[3]},{q_f[0]},{q_f[1]},{q_f[2]}")
        data_rows.append(row)

    # (Escritura del archivo... igual que siempre)
    with open(output_path, 'w') as f:
        f.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.5\n")
        f.write(f"nRows={len(data_rows)}\nnColumns={len(dict_dfs) + 1}\n")
        f.write("endheader\n")
        pd.DataFrame(data_rows, columns=['time'] + list(dict_dfs.keys())).to_csv(f, sep='\t', index=False)

# Llamadas finales
export_sto(dict_dfs, sto_movement, segment_ranges, s2s_matrices)
export_sto(dict_dfs, sto_calibration, segment_ranges, s2s_matrices)

# Rotación ENU→OpenSim
R_enu_to_osim = R.from_euler('x', -np.pi/2)

# Orientación estática de pelvis en OpenSim
r_est_p = segment_ranges['pelvis_imu'][0]
q_pelvis = R.from_quat(
    dict_dfs['pelvis_imu'].iloc[r_est_p[0]:r_est_p[1]][q_cols_xsens].dropna().values
).mean()
s2s_pelvis = s2s_matrices['pelvis_imu']

q_anat_osim = R_enu_to_osim * q_pelvis * s2s_pelvis
mtx = q_anat_osim.as_matrix()

print("Ejes de pelvis en espacio OpenSim:")
print(f"  X (debe ser frente del modelo): {mtx[:,0].round(2)}")
print(f"  Y (debe ser arriba):            {mtx[:,1].round(2)}")
print(f"  Z (debe ser lateral):           {mtx[:,2].round(2)}")

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

R_enu_to_osim = R.from_euler('x', -np.pi / 2)

r_est_p = segment_ranges['pelvis_imu'][0]
q_pelvis = R.from_quat(
    dict_dfs['pelvis_imu'].iloc[r_est_p[0]:r_est_p[1]][q_cols_xsens].dropna().values
).mean()
s2s_pelvis = s2s_matrices['pelvis_imu']

# Orientación del sensor de pelvis en OpenSim (sin corrección de heading todavía)
q_sensor_osim = R_enu_to_osim * q_pelvis * s2s_pelvis
mtx = q_sensor_osim.as_matrix()

ejes = {
    'x':  mtx[:, 0],
    '-x': -mtx[:, 0],
    'y':  mtx[:, 1],
    '-y': -mtx[:, 1],
    'z':  mtx[:, 2],
    '-z': -mtx[:, 2],
}

# El frente en OpenSim es +X = [1, 0, 0]
frente_osim = np.array([1.0, 0.0, 0.0])

print("¿Qué eje del sensor apunta más al frente del modelo en OpenSim?")
for nombre, vec in ejes.items():
    # Proyectamos en el plano horizontal de OpenSim (ignoramos Y)
    vec_horiz = np.array([vec[0], 0.0, vec[2]])
    if np.linalg.norm(vec_horiz) > 0:
        vec_horiz /= np.linalg.norm(vec_horiz)
    dot = np.dot(vec_horiz, frente_osim)
    print(f"  {nombre:>3}:  componente X = {vec[0]:+.3f}   dot con frente = {dot:+.3f}")

print("\n" + "="*50)
print("="*50)
