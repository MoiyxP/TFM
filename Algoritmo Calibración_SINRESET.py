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
# Gráficas validación
# ------------------------------------------------------------------

def plot_gyro_scatter_pca(gyr_data, data_transformed, seg_name, pca, graphs_folder):
    """
    Visualización de la nube de puntos. 
    Si pca es None, asume que data_transformed ya viene en el espacio del segmento (X, Y, Z).
    """
    # Configuración de etiquetas según el método
    if pca is not None:
        var = pca.explained_variance_ratio_ * 100
        titulo = (f'Validación PCA: {seg_name}\n'
                  f'PC1={var[0]:.1f}%  PC2={var[1]:.1f}%  PC3={var[2]:.1f}%')
        label_x, label_y, label_z = 'PC1 (Flex/Ext)', 'PC2', 'PC3'
    else:
        titulo = f'Validación Funcional (Aceleración): {seg_name}'
        label_x, label_y, label_z = 'X (Fwd)', 'Y (Up)', 'Z (ML)'

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(titulo, fontsize=14)
    mag = np.linalg.norm(gyr_data, axis=1)

    # --- Fila 1: Vistas 3D ---
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(gyr_data[:,0], gyr_data[:,1], gyr_data[:,2], c=mag, cmap='viridis', alpha=0.5, s=8)
    ax1.set_title('ANTES (Sensor)')
    
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(data_transformed[:,0], data_transformed[:,1], data_transformed[:,2], c=mag, cmap='viridis', alpha=0.5, s=8)
    ax2.set_title('DESPUÉS (Segmento)')
    ax2.set_xlabel(label_x); ax2.set_ylabel(label_y); ax2.set_zlabel(label_z)

    # --- Fila 2: Proyecciones 2D ---
    # Plano Sagital (X-Y o PC1-PC2)
    ax3 = fig.add_subplot(2, 3, 4)
    ax3.scatter(data_transformed[:,0], data_transformed[:,1], c=mag, cmap='viridis', alpha=0.6, s=8)
    ax3.set_xlabel(label_x); ax3.set_ylabel(label_y)
    ax3.set_title('Plano Sagital/Frontal')

    # Plano Horizontal (X-Z o PC1-PC3)
    ax4 = fig.add_subplot(2, 3, 5)
    ax4.scatter(data_transformed[:,0], data_transformed[:,2], c=mag, cmap='viridis', alpha=0.6, s=8)
    ax4.set_xlabel(label_x); ax4.set_ylabel(label_z)
    ax4.set_title('Plano Horizontal')

    # Plano Frontal (Y-Z o PC2-PC3)
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.scatter(data_transformed[:,1], data_transformed[:,2], c=mag, cmap='viridis', alpha=0.6, s=8)
    ax5.set_xlabel(label_y); ax5.set_ylabel(label_z)
    ax5.set_title('Plano Frontal')

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        if hasattr(ax, 'set_zlim'): ax.set_zlim([-6,6])
        ax.set_xlim([-6,6]); ax.set_ylim([-6,6])
        if isinstance(ax, plt.Axes): ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(graphs_folder, f'Validacion_{seg_name}.png'), dpi=150)
    plt.close()

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

def calibrate_segment_functional(df, r_est, r_mov, seg, q_cols):
    # === PASO 1: EJE VERTICAL (Y) ===
    q_static = df.iloc[r_est[0]:r_est[1]][q_cols].dropna().values
    rot_avg  = R.from_quat(q_static).mean()
    v_vert   = rot_avg.inv().apply([0, 0, 1])
    v_vert  /= np.linalg.norm(v_vert)

    # === PASO 2: EJE DE FLEXIÓN (PCA -> Z) ===
    gyr_cols = get_cols(df, 'Gyr')
    gyr_data = df.iloc[r_mov[0]:r_mov[1]][gyr_cols].dropna().values
    pca = PCA(n_components=3).fit(gyr_data)
    v_flex = pca.components_[0]
    
    # Ortogonalizar Z respecto a Y
    v_flex = v_flex - np.dot(v_flex, v_vert) * v_vert
    v_flex /= np.linalg.norm(v_flex)

    # === PASO 3: EJE FORWARD (X) ===
    v_fwd = np.cross(v_vert, v_flex) # Y x Z = X
    v_fwd /= np.linalg.norm(v_fwd)

    # --- CORRECCIÓN DE SIGNO ---
    acc_cols = get_cols(df, 'FreeAcc_') or get_cols(df, 'Acc_')
    acc_mean = np.mean(df.iloc[r_mov[0]:r_mov[0]+100][acc_cols].values, axis=0)
    
    # Si X apunta atrás, invertimos X y Z (Y se queda igual)
    if np.dot(v_fwd, acc_mean) < 0:
        v_fwd = -v_fwd
        v_flex = -v_flex

    mat = np.column_stack((v_fwd, v_vert, v_flex))
    det = np.linalg.det(mat)
    assert abs(det - 1.0) < 1e-4, f"{seg}: Error de determinante {det}"

    return R.from_matrix(mat), pca, gyr_data, v_vert, v_fwd, v_flex

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

### ------------- BUCLE DE CALIBRACIÓN HÍBRIDO

segment_ranges = {}
s2s_matrices = {} 
calib_log = []
q_cols_xsens = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']

for seg, df in dict_dfs.items():
    # 1. Segmentar periodos
    r_est, r_mov = segment_data(df, seg, idx_turn_global)
    segment_ranges[seg] = (r_est, r_mov)

    if seg == 'pelvis_imu':
        print(f"[{seg}] Calibración funcional por aceleración...")
        q_static = df.iloc[r_est[0]:r_est[1]][q_cols_xsens].dropna().values
        rot_avg = R.from_quat(q_static).mean()
        v_v = rot_avg.inv().apply([0, 0, 1])
        v_v /= np.linalg.norm(v_v)
        
        v_m = get_ml_functional_acc(df, r_mov, v_v) # Z (ML)
        v_f = np.cross(v_v, v_m)                   # X (Fwd)
        v_f /= np.linalg.norm(v_f)
        
        mat = np.column_stack((v_f, v_v, v_m))
        det = np.linalg.det(mat)
        if det < 0:
            v_m = -v_m
            mat = np.column_stack((v_f, v_v, v_m))
        
        s2s = R.from_matrix(mat)
        pca_obj = None
        var_pc1_display = "Funcional"
        desv_deg = 90.0
        gyr_data = df.iloc[r_mov[0]:r_mov[1]][get_cols(df, 'Gyr')].dropna().values
        data_plot = s2s.inv().apply(gyr_data)

    else:
        # === MIEMBROS: PCA funcional ===
        s2s, pca_obj, gyr_data, v_v, v_f, v_m = calibrate_segment_functional(df, r_est, r_mov, seg, q_cols_xsens)
        data_plot = pca_obj.transform(gyr_data)
        
        # Variables para log
        var_pc1_display = f"{pca_obj.explained_variance_ratio_[0]*100:.1f}%"
        dot_prod = np.dot(v_v, v_f)
        desv_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))

    # 3. Guardar resultados
    s2s_matrices[seg] = s2s
    
    # 4. IMPRIMIR LOG (Formato recuperado)
    print(f"[{seg}] PC1 (Z-ML) Var: {var_pc1_display} | Angulo V-Z: 90.00°")

    # 5. Gráfica de validación
    plot_gyro_scatter_pca(gyr_data, data_plot, seg, pca_obj, graphs_folder)

    # 6. Guardar en el Excel
    calib_log.append({
        'Segmento': seg,
        'Metodo': 'PCA' if pca_obj else 'Aceleracion',
        'PC1_Variance_%': var_pc1_display,
        'Angulo_V_X_Deg': desv_deg,
        'V_Vert_X': v_v[0], 'V_Vert_Y': v_v[1], 'V_Vert_Z': v_v[2],
        'V_Flex_X': v_f[0], 'V_Flex_Y': v_f[1], 'V_Flex_Z': v_f[2]
    })

pd.DataFrame(calib_log).to_excel(final_excel, index = False)

# ------------------------------------------------------------------
# Exportación OpenSim
# ------------------------------------------------------------------

### ------------- Archivos .STO

def export_sto(dict_dfs, output_path, segment_ranges, s2s_matrices):
    min_len = min([len(df) for df in dict_dfs.values()])
    q_cols = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']
    is_static = "calibration" in output_path.lower()

    # HEADING INDEPENDIENTE POR SEGMENTO
    dict_corrections = {}
    for seg, df in dict_dfs.items():
        t_st = segment_ranges[seg][0]
        q_raw_st = R.from_quat(
            df.iloc[t_st[0]:t_st[1]][q_cols].dropna().values
        ).mean()
        q_anat_world = q_raw_st * s2s_matrices[seg]
        fwd_vec = q_anat_world.apply([1, 0, 0])
        fwd_vec[2] = 0
        fwd_vec /= np.linalg.norm(fwd_vec)
        yaw_angle = np.arctan2(fwd_vec[1], fwd_vec[0])
        dict_corrections[seg] = R.from_euler('z', -yaw_angle)
        print(f"  {seg}: Fwd={np.round(fwd_vec,3)} Yaw={np.degrees(yaw_angle):.1f}°")

    data_rows = []
    frames = 1 if is_static else min_len

    for i in range(frames):
        row = [i / 100.0]
        for seg, df in dict_dfs.items():
            if is_static:
                t = segment_ranges[seg][0]
                q_raw = R.from_quat(
                    df.iloc[t[0]:t[1]][q_cols].dropna().values
                ).mean()
            else:
                q_raw = R.from_quat(df.iloc[i][q_cols].values)

            r_f = dict_corrections[seg] * (q_raw * s2s_matrices[seg])
            q_f = r_f.as_quat()
            if q_f[3] < 0:
                q_f = -q_f
            row.append(f"{q_f[3]},{q_f[0]},{q_f[1]},{q_f[2]}")
        data_rows.append(row)

    with open(output_path, 'w') as f:
        f.write("DataRate=100.000000\nDataType=Quaternion\nversion=3\n"
                "OpenSimVersion=4.5\nendheader\n")
        pd.DataFrame(
            data_rows,
            columns=['time'] + list(dict_dfs.keys())
        ).to_csv(f, sep='\t', index=False)

    nombre = "ESTÁTICO" if is_static else "MOVIMIENTO"
    print(f"Archivo exportado: {os.path.basename(output_path)} ({nombre})")

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