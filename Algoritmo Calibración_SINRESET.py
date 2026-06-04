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

data_adress = r"C:\Users\Víctor\Documents\Estudios\Prácticas Biomech\Proyecto_Xsens\Archivos\DATOS_VICTOR\MARCHA_IDAVUELTA_SENSORES_ALINEADOS_SINRESET"
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

def get_ml_functional_acc(df, t_mov, v_vert_local):
    # Definimos una ventana corta al inicio (100 frames) para evitar la vuelta
    t_inicio_marcha = t_mov[0]
    t_fin_referencia = min(t_mov[0] + 100, t_mov[1]) 
    
    acc_cols = get_cols(df,'FreeAcc_') or get_cols(df,'Acc_')
    # CALCULAMOS LA MEDIA SOLO DEL PRINCIPIO (Ida)
    v_acc_mean = np.mean(df.iloc[t_inicio_marcha : t_fin_referencia][acc_cols].values, axis = 0)

    # Proyectamos horizontalmente el eje (esto quita los 45 grados)
    v_fwd = v_acc_mean - np.dot(v_acc_mean, v_vert_local) * v_vert_local
    v_fwd /= np.linalg.norm(v_fwd)

    # Sacamos el eje mediolateral, perpendicular a la vertical y el avance
    v_ml = np.cross(v_vert_local, v_fwd)
    return v_ml / np.linalg.norm(v_ml)

def get_orthonormal_basis(v_vert, v_ml):
    """
    Construye la base: 
    - Z = Vertical
    - Y = ML
    - X = Forward
    """
    z_axis = v_vert / np.linalg.norm(v_vert)
    y_axis = v_ml - np.dot(v_ml, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    return R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))

### ------------- BUCLE DE CALIBRACIÓN

segment_ranges = {}
s2s_matrices = {} # Almacenaje de la rotación sensor-segmento
calib_log = []
q_cols_xsens = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0',]

for seg, df in dict_dfs.items():
    r_est, r_mov = segment_data(df, seg, idx_turn_global)
    segment_ranges[seg] = (r_est, r_mov)

    # Calibración Vertical (Buscamos la gravedad durante la N-Pose)
    q_static = df.iloc[r_est[0]:r_est[1]][q_cols_xsens].dropna().values # Usamos los cuaterniones
    rot_avg_static = R.from_quat(q_static).mean() # Buscamos la orientación media que mejor representa a todas las muestras.
    # Si usásemos np.mean, tendríamos un problema por los signos negativos
    v_vert = rot_avg_static.inv().apply([0, 0, 1]) #Determinamos la dirección de la gravedad desde la perspectiva del sensor
    # Este será nuestro vector vertical de referencia. 

    # Calibración Horizontal (Eje ML -> PCA usando el giroscopio)
    if seg == 'pelvis_imu':
        v_ml = get_ml_functional_acc(df, r_mov, v_vert)
    else:    
        gyr_data = df.iloc[r_mov[0]:r_mov[1]][get_cols(df, 'Gyr')].dropna().values
        v_ml = PCA(n_components = 1).fit(gyr_data).components_[0]

    # Construimos la base S2S -> Matriz que alinea el sensor con el hueso
    s2s_matrices[seg] = get_orthonormal_basis(v_vert, v_ml)

    # Datos para el excel
    dot_product = np.clip(np.dot(v_vert, v_ml), -1.0, 1.0)
    desv_deg = abs(90 - np.degrees(np.arccos(dot_product)))

    calib_log.append({
        'Segmento': seg,
        'Desviación_Ortogonalidad_Deg': desv_deg,
        'V_Vert_X': v_vert[0], 'V_Vert_Y': v_vert[1], 'V_Vert_Z': v_vert[2],
        'V_ML_X': v_ml[0], 'V_ML_Y': v_ml[1], 'V_ML_Z': v_ml[2]
    })

pd.DataFrame(calib_log).to_excel(final_excel, index = False)

# ------------------------------------------------------------------
# Exportación OpenSim
# ------------------------------------------------------------------

### ------------- Archivos .STO

def export_sto(dict_dfs, output_path, segment_ranges, s2s_matrices):
    min_len = min([len(df) for df in dict_dfs.values()])
    q_cols = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']
    
    # --- PASO A: ALINEACIÓN DE HEADING POR SEGMENTO ---
    # Al usar arctan2, el código detecta automáticamente si el sensor
    # mira hacia adelante o hacia atrás y lo corrige a 0º (X+).
    dict_corrections = {}
    for seg, df in dict_dfs.items():
        t_st = segment_ranges[seg][0]
        q_raw_st = R.from_quat(df.iloc[t_st[0]:t_st[1]][q_cols].dropna().values).mean()
        
        # Orientación anatómica inicial en el mundo
        q_anat_world = q_raw_st * s2s_matrices[seg]
        
        # Dirección del eje X (adelante) de este hueso
        fwd_vec = q_anat_world.apply([1, 0, 0])
        
        # Ángulo de desviación respecto al eje X del laboratorio
        yaw_angle = np.arctan2(fwd_vec[1], fwd_vec[0])
        
        # Aplicamos la corrección de Yaw. 
        # Como tu pelvis ya está calibrada con el avance, esto es infalible.
        dict_corrections[seg] = R.from_euler('z', -yaw_angle)

    # --- PASO B: GENERACIÓN DE DATOS ---
    data_rows = []
    is_static = "calibration" in output_path

    for i in range(1 if is_static else min_len):
        row = [i/100.0]
        for seg, df in dict_dfs.items():
            if is_static:
                t = segment_ranges[seg][0]
                q_raw = R.from_quat(df.iloc[t[0]:t[1]][q_cols].dropna().values).mean()
            else:
                q_raw = R.from_quat(df.iloc[i][q_cols].values)

            # Transformación: Corrección_Yaw * (Movimiento * Calibración_Anatómica)
            r_f = dict_corrections[seg] * (q_raw * s2s_matrices[seg])
            q_f = r_f.as_quat()

            # Estabilidad de signos (W positiva)
            if q_f[3] < 0: q_f = -q_f

            # Formato OpenSim [w, x, y, z]
            row.append(f"{q_f[3]},{q_f[0]},{q_f[1]},{q_f[2]}")

        data_rows.append(row)

    # Escritura del archivo
    with open(output_path, 'w') as f:
        f.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\nOpenSimVersion=4.5\nendheader\n")
        pd.DataFrame(data_rows, columns = ['time'] + list(dict_dfs.keys())).to_csv(f, sep = '\t', index = False)

export_sto(dict_dfs, sto_movement, segment_ranges, s2s_matrices)
export_sto(dict_dfs, sto_calibration, segment_ranges, s2s_matrices)

### ------------- Archivo XML
root = ET.Element("OpenSimDocument", Version = "40000")
placer = ET.SubElement(root, "IMUPlacer")
ET.SubElement(placer, "base_imu_label").text = 'pelvis_imu'
ET.SubElement(placer, "base_heading_axis").text = 'x'
ET.SubElement(placer, "sensor_to_opensim_rotations").text = "-1.5707963267948966 0 0"
ET.SubElement(placer, "orientation_file_for_calibration").text = os.path.basename(sto_calibration)
with open(xml_placer, 'w') as f: f.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="   "))

print(f"\n>>> PROCESO FINALIZADO CON ÉXITO")