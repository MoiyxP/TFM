import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
from xml.dom import minidom
from mpl_toolkits.mplot3d import Axes3D

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
# Gráficas de validación
# ------------------------------------------------------------------

def plot_gyro_scatter(df, seg_name, s2s_matrix, motion_range, graphs_folder):
    """
    Crea un Scatter Plot 3D de la velocidad angular para visualizar
    el plano de rotación antes y después de la calibración.
    """
    # 1. Extraer datos (Raw y Segmento)
    gyr_cols = get_cols(df, 'Gyr')
    data_raw = df.iloc[motion_range[0]:motion_range[1]][gyr_cols].values
    
    # Transformamos al espacio del segmento (usando la misma lógica que el anterior)
    inv_s2s = s2s_matrix.as_matrix().T
    data_segment = s2s_matrix.inv().apply(data_raw)

    # 2. Configurar el gráfico
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f'Nube de Rotación (Scatter 3D): {seg_name}', fontsize=16)

    # SUBPLOT 1: Espacio SENSOR (El "Caos" inicial)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data_raw[:, 0], data_raw[:, 1], data_raw[:, 2], c=data_raw[:, 2], cmap='coolwarm', alpha=0.6)
    ax1.set_title('ANTES (Espacio Sensor)')
    ax1.set_xlabel('Gyr X'); ax1.set_ylabel('Gyr Y'); ax1.set_zlabel('Gyr Z')

    # SUBPLOT 2: Espacio SEGMENTO (La "Alineación" final)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(data_segment[:, 0], data_segment[:, 1], data_segment[:, 2], c=data_segment[:, 0], cmap='coolwarm', alpha=0.6)
    ax2.set_title('DESPUÉS (Espacio Segmento)')
    ax2.set_xlabel('Hueso X (Fwd)'); ax2.set_ylabel('Hueso Y (Up)'); ax2.set_zlabel('Hueso Z (ML)')

    # Normalizar ejes para que la comparación sea justa
    for ax in [ax1, ax2]:
        ax.set_xlim([-6, 6]); ax.set_ylim([-6, 6]); ax.set_zlim([-6, 6])

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_folder, f'Gyro_Scatter_{seg_name}.png'))
    plt.close()

def plot_gyro_validation(df, seg_name, s2s_matrix, motion_range, graphs_folder):
    """
    Genera una comparativa de la Velocidad Angular en 3 sistemas:
    1. Sensor (Raw): Datos tal cual los lee el chip.
    2. Segmento (Hueso): Datos corregidos por la matriz S2S.
    3. Global (Lab): Datos orientados respecto al suelo.
    """
    # 1. Extraer datos del giroscopio (solo fase de marcha)
    gyr_cols = get_cols(df, 'Gyr')
    data_raw = df.iloc[motion_range[0]:motion_range[1]][gyr_cols].values
    
    # 2. Transformar al Sistema del Segmento (Hueso)
    # Aplicamos la rotación de la matriz S2S a cada vector de velocidad angular
    # para pasar del espacio del SENSOR al espacio del SEGMENTO (Hueso)
    data_segment = s2s_matrix.inv().apply(data_raw)

    # 3. Transformar al Sistema Global (Mundo)
    # Usamos los cuaterniones del sensor para ver la rotación en el lab
    q_cols = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']
    quats = R.from_quat(df.iloc[motion_range[0]:motion_range[1]][q_cols].values)
    data_global = quats.apply(data_raw)

    # --- CÁLCULO DE PC1 y PC2 (Para el informe) ---
    pca = PCA(n_components=3)
    pca.fit(data_raw)
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # --- PLOT ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Validación Funcional: {seg_name}\nPC1 (Flex/Ext): {var_pc1:.1f}% var. | PC2 (Abd/Add): {var_pc2:.1f}% var.', fontsize=14)

    time_axis = np.arange(len(data_raw))
    labels = ['Eje X', 'Eje Y', 'Eje Z']
    colors = ['#ff4b4b', '#2ecc71', '#00a8ff'] # Rojo, Verde, Azul

    # Gráfica 1: Sensor (Todo mezclado)
    for i in range(3):
        axs[0].plot(time_axis, data_raw[:, i], color=colors[i], label=labels[i], alpha=0.8)
    axs[0].set_title('Espacio SENSOR (Raw Data - Antes de Calibrar)')
    axs[0].legend(loc='upper right')

    # Gráfica 2: Segmento (¡Aquí está la magia!)
    # Si está bien, uno de los ejes (PC1) debe absorber casi todo el movimiento
    for i in range(3):
        axs[1].plot(time_axis, data_segment[:, i], color=colors[i], label=labels[i])
    axs[1].set_title('Espacio SEGMENTO (Anatómico - Después de Calibrar)')
    axs[1].set_ylabel('Velocidad Angular (rad/s)')

    # Gráfica 3: Global (Referencia Laboratorio)
    for i in range(3):
        axs[2].plot(time_axis, data_global[:, i], color=colors[i], label=labels[i])
    axs[2].set_title('Espacio GLOBAL (Referencia Laboratorio)')
    axs[2].set_xlabel('Frames')

    for ax in axs: 
        ax.grid(alpha=0.3)
        ax.set_ylim([-6, 6]) # Ajustar según intensidad de marcha

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(graphs_folder, f'Gyro_Val_{seg_name}.png'))
    plt.close()


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

def get_ml_functional_acc(df, t_mov, v_vert):
    t_fin_ref = min(t_mov[0] + 100, t_mov[1])
    
    # Priorizar FreeAcc_ sobre Acc_
    acc_cols = get_cols(df, 'FreeAcc_') or get_cols(df, 'Acc_')
    
    v_acc_mean = np.mean(df.iloc[t_mov[0]:t_fin_ref][acc_cols].values, axis=0)
    
    # Eliminamos componente gravitacional residual proyectando al plano horizontal
    v_fwd = v_acc_mean - np.dot(v_acc_mean, v_vert) * v_vert
    norm = np.linalg.norm(v_fwd)
    
    if norm < 0.01:
        print("  ⚠️ pelvis: Aceleración demasiado pequeña. Usando fallback.")
        return np.array([0.0, 1.0, 0.0])
    
    v_fwd /= norm
    # ML = Vertical x Forward
    v_ml = np.cross(v_vert, v_fwd)
    return v_ml / np.linalg.norm(v_ml)

def get_orthonormal_basis(v_vert, v_ml):
    # v_ml ya está ortogonalizado respecto a v_vert
    y_axis = v_vert / np.linalg.norm(v_vert)
    z_axis = v_ml / np.linalg.norm(v_ml)
    x_axis = np.cross(y_axis, z_axis)  # mano derecha: Up × ML = Fwd
    x_axis /= np.linalg.norm(x_axis)

    # Verificación
    det = np.linalg.det(np.column_stack((x_axis, y_axis, z_axis)))
    assert abs(det - 1.0) < 1e-4, f"{seg}: determinante = {det:.6f}"

    return R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))

### ------------- ESTIMACIÓN DE RUMBO (HEADING) GLOBAL
# El sensor de pelvis está en la ESPALDA -> su Yaw apunta hacia atrás.
# Añadimos 180° para que el Forward apunte hacia adelante del sujeto.
yaw_inicio = dict_dfs['pelvis_imu']['Yaw'].iloc[20:120].mean()
print(f"\n>>> CONFIGURACIÓN GLOBAL:")
print(f"Yaw inicial pelvis (espalda): {yaw_inicio:.1f}°")

heading_rad = np.deg2rad(yaw_inicio + 180) 
forward_global = np.array([np.cos(heading_rad), np.sin(heading_rad), 0])
right_global_enu = np.array([np.sin(heading_rad), -np.cos(heading_rad), 0])

print(f"Forward global ENU (corregido): {np.round(forward_global, 3)}")
print(f"Right global ENU   (corregido): {np.round(right_global_enu, 3)}")

### ------------- BUCLE DE CALIBRACIÓN UNIFICADO (CORRECCIÓN ANATÓMICA + PRINTS)

segment_ranges = {}
s2s_matrices = {} 
calib_log = []
q_cols_xsens = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']

for seg, df in dict_dfs.items():
    print(f"\n" + "="*50)
    print(f">>> PROCESANDO SEGMENTO: {seg}")

    # 1. SEGMENTACIÓN
    r_est, r_mov = segment_data(df, seg, idx_turn_global)
    segment_ranges[seg] = (r_est, r_mov)

    # 2. CALIBRACIÓN VERTICAL (Eje Y OpenSim / Up)
    q_static_data = df.iloc[r_est[0]:r_est[1]][q_cols_xsens].dropna().values
    if len(q_static_data) == 0:
        print(f"⚠️ {seg}: Sin datos estáticos suficientes.")
        continue
    
    rot_avg_static = R.from_quat(q_static_data).mean()
    v_vert = rot_avg_static.inv().apply([0, 0, 1]) # Gravedad en el sensor
    v_vert /= np.linalg.norm(v_vert)

    # 3. CALIBRACIÓN HORIZONTAL / ML (Eje Z OpenSim / ML)
    if seg == 'pelvis_imu':
        # La pelvis usa aceleración funcional
        v_ml = get_ml_functional_acc(df, r_mov, v_vert)
    else:
        # Miembros usan PCA del giroscopio
        gyr_data = df.iloc[r_mov[0]:r_mov[1]][get_cols(df, 'Gyr')].dropna().values
        pca = PCA(n_components=3).fit(gyr_data)
        
        # Seleccionamos el componente de rotación más horizontal (bisagra)
        best_pc = None
        best_norm = 0
        for pc in pca.components_:
            proj = pc - np.dot(pc, v_vert) * v_vert
            if np.linalg.norm(proj) > best_norm:
                best_norm = np.linalg.norm(proj)
                best_pc = proj / best_norm
        v_ml = best_pc

    # --- CORRECCIÓN DE SIGNO POR ANATOMÍA ---
    # Proyectamos la "Derecha Global" (basada en el heading + 180) al sensor
    right_in_sensor = rot_avg_static.inv().apply(right_global_enu)
    right_proj = right_in_sensor - np.dot(right_in_sensor, v_vert) * v_vert
    right_proj /= np.linalg.norm(right_proj)

    # Target: Z apunta a la IZQUIERDA en pierna izquierda, DERECHA en el resto
    if '_l_imu' in seg:
        target_ml = -right_proj 
    else:
        target_ml = right_proj  

    # Comprobación de signo de la PCA vs Anatomía
    dot_anat = np.dot(v_ml, target_ml)
    if dot_anat < 0:
        v_ml = -v_ml
        print(f"  [AVISO] v_ml invertido para coincidir con lado {('Izquierdo' if '_l_imu' in seg else 'Derecho')}")
    
    # 4. BASE ORTONORMAL (X=Fwd, Y=Up, Z=ML)
    y_axis = v_vert
    z_axis = v_ml
    x_axis = np.cross(y_axis, z_axis) # Mano derecha: Up x ML = Fwd
    x_axis /= np.linalg.norm(x_axis)

    s2s_matrices[seg] = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))

    # 5. DIAGNÓSTICOS DE CONSOLA (MANTENIDOS)
    mat = s2s_matrices[seg].as_matrix()
    print(f"  - Eje X (Fwd) local: {np.round(mat[:,0], 3)}")
    print(f"  - Eje Y (Up)  local: {np.round(mat[:,1], 3)}")
    print(f"  - Eje Z (ML)  local: {np.round(mat[:,2], 3)}")
    print(f"  - Determinante:      {np.linalg.det(mat):.6f}")
    print(f"  - Dot Anatómico:     {dot_anat:.3f} (Debe ser > 0)")
    print(f"  - Ortogonalidad Y-Z: {np.dot(y_axis, z_axis):.4f} (Debe ser ~0)")
    print(f"  - ML en plano Horiz: {np.linalg.norm(v_ml[:2]):.3f}")
    
    # Datos de los cuaterniones usados
    print(f"  - Frames estáticos:  {len(q_static_data)}")
    print(f"  - Varianza Quat:     {np.round(np.var(q_static_data, axis=0), 6)}")

    # 6. GRÁFICAS DE VALIDACIÓN
    plot_gyro_validation(df, seg, s2s_matrices[seg], r_mov, graphs_folder)
    plot_gyro_scatter(df, seg, s2s_matrices[seg], r_mov, graphs_folder)

    # LOG PARA EXCEL
    calib_log.append({
        'Segmento': seg,
        'Dot_Anatomico': dot_anat,
        'V_Vert': v_vert.tolist(),
        'V_ML': v_ml.tolist()
    })

pd.DataFrame(calib_log).to_excel(final_excel, index=False)

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
