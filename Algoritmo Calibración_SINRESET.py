import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy.signal import butter, filtfilt

#### Rutas carpetas y datos

data_adress = r"C:\Users\Víctor\Documents\Estudios\Prácticas Biomech\Proyecto_Xsens\Archivos\DATOS_VICTOR\MARCHA_IDA_SENSORES_ALINEADOS_SINRESET"
results_rout = r"C:\Users\Víctor\Documents\Estudios\Prácticas Biomech\Proyecto_Xsens\Resultados"

final_excel = os.path.join(results_rout, "Excel_Datos.xlsx")
pca_graphs_folder = os.path.join(results_rout, "Gráficas_PCA")   
graphs_folder = os.path.join(results_rout, "Gráficas_Segmentación")
sto_movement = os.path.join(results_rout, "march_movement.sto")
sto_calibration = os.path.join(results_rout, "static_calibration.sto")
xml_placer = os.path.join(results_rout, "configuration_placer.xml")

if not os.path.exists(graphs_folder): os.makedirs(graphs_folder)
if not os.path.exists(pca_graphs_folder): os.makedirs(pca_graphs_folder)

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

    return static_range, motion_range, fs

# ------------------------------------------------------------------
# Gráficas validación
# ------------------------------------------------------------------

def plot_sensor_orientation_in_segment(s2s_matrices, pca_graphs_folder):
    """
    Muestra cómo están orientados los ejes físicos del SENSOR (X,Y,Z)
    dentro del sistema de coordenadas del HUESO.
    Si los sensores están 'desordenados', aquí se verá claramente.
    """
    segments = list(s2s_matrices.keys())
    n_seg = len(segments)
    cols = 3
    rows = (n_seg // cols) + (1 if n_seg % cols > 0 else 0)
    
    fig = plt.figure(figsize=(15, rows * 5))
    fig.suptitle('Orientación Física del SENSOR dentro de cada HUESO\n'
                 '(Ejes del chip: Rojo=X_sens, Verde=Y_sens, Azul=Z_sens)', fontsize=18)

    sensor_colors = ['#ff4b4b', '#2ecc71', '#00a8ff'] # Colores de los ejes del sensor

    for i, seg in enumerate(segments):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        # Extraemos la matriz S2S (Sensor -> Segmento)
        # Las columnas de esta matriz son los vectores unitarios del sensor 
        # expresados en el sistema del hueso.
        mat_s2s = s2s_matrices[seg].as_matrix()

        # 1. Dibujamos el sistema del HUESO como referencia fija (Caja gris)
        # Eje X_hueso (Fwd), Y_hueso (Up), Z_hueso (ML)
        ax.quiver(0, 0, 0, 1, 0, 0, color='black', alpha=0.2, lw=2, linestyle='-') 
        ax.quiver(0, 0, 0, 0, 1, 0, color='black', alpha=0.2, lw=2, linestyle='-')
        ax.quiver(0, 0, 0, 0, 0, 1, color='black', alpha=0.2, lw=2, linestyle='-')

        # 2. Dibujamos los ejes reales del SENSOR (X, Y, Z del chip)
        # Estos son los que cambian si 'desordenas' los sensores
        for axis_idx in range(3):
            # La columna i de la matriz es la dirección del eje i del sensor
            vec = mat_s2s[:, axis_idx] 
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                      color=sensor_colors[axis_idx], linewidth=3, length=0.8)

        ax.set_title(f'SENSOR en {seg.upper()}', fontweight='bold')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(pca_graphs_folder, 'Orientacion_Sensores_Interna.png'), dpi=150)
    plt.close()

def plot_calibration_evolution(gyr_raw, s2s_initial, s2s_final, seg_name, pca_graphs_folder):
    """
    Compara tres estados de la señal:
    1. Cruda (Sensor)
    2. Funcional (PCA pura, puede estar invertida)
    3. Alineada (Tras corrección con Pelvis, lista para OpenSim)
    """
    # Transformamos los datos
    data_pca_local = s2s_initial.inv().apply(gyr_raw)
    data_final = s2s_final.inv().apply(gyr_raw)
    
    time = np.arange(len(gyr_raw)) / 100.0
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'Evolución de la Calibración: {seg_name}', fontsize=16, fontweight='bold')

    colors = ['#ff4b4b', '#2ecc71', '#00a8ff'] # X, Y, Z (o PC1, 2, 3)

    # PANEL 1: SEÑAL CRUDA (SENSOR)
    for i in range(3):
        axs[0].plot(time, gyr_raw[:, i], color=colors[i], alpha=0.6)
    axs[0].set_title('1. ESPACIO SENSOR (Datos brutos mezclados)')
    axs[0].set_ylabel('rad/s')

    # PANEL 2: PCA LOCAL (PRE-ALINEACIÓN)
    # Aquí PC1 (Z) ya domina, pero podría estar apuntando "hacia el lado equivocado"
    axs[1].plot(time, data_pca_local[:, 2], color='blue', linewidth=2, label='Z (Flexión PCA)')
    axs[1].plot(time, data_pca_local[:, 0], color='red', alpha=0.4, label='X (Fwd)')
    axs[1].set_title('2. PCA FUNCIONAL (Movimiento aislado, rumbo sin verificar)')
    axs[1].legend(loc='upper right')

    # PANEL 3: ALINEACIÓN GLOBAL (POST-PELVIS)
    # Aquí el signo de la flexión es consistente en todo el esqueleto
    axs[2].plot(time, data_final[:, 2], color='blue', linewidth=2, label='Z (Flexión Final)')
    axs[2].plot(time, data_final[:, 0], color='red', alpha=0.4, label='X (Fwd Final)')
    axs[2].set_title('3. SISTEMA ALINEADO (Corrección de 180° aplicada si era necesaria)')
    axs[2].set_ylabel('rad/s')
    axs[2].set_xlabel('Tiempo (s)')
    axs[2].legend(loc='upper right')

    for ax in axs: ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(pca_graphs_folder, f'Calibration_Evolution_{seg_name}.png'), dpi=150)
    plt.close()

def plot_axes_gallery(dict_dfs, s2s_matrices, segment_ranges, dict_corrections, pca_graphs_folder):
    """
    Crea una galería de subplots 3D mostrando el sistema de coordenadas
    final de cada hueso en relación al laboratorio (X+ hacia adelante).
    """
    segments = list(s2s_matrices.keys())
    n_seg = len(segments)
    
    # Configuramos la cuadrícula (ej. 3 columnas)
    cols = 3
    rows = (n_seg // cols) + (1 if n_seg % cols > 0 else 0)
    
    fig = plt.figure(figsize=(15, rows * 5))
    fig.suptitle('Sistemas de Coordenadas Finales (Pose Estática)\nRojo=X(Fwd), Verde=Y(Up), Azul=Z(ML)', fontsize=18)

    q_cols = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']
    colors = ['#ff4b4b', '#2ecc71', '#00a8ff'] # X, Y, Z

    for i, seg in enumerate(segments):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        # 1. Obtener la orientación estática del sensor
        t_st = segment_ranges[seg][0]
        q_raw_st = R.from_quat(dict_dfs[seg].iloc[t_st[0]:t_st[1]][q_cols].dropna().values).mean()
        
        # 2. Aplicar la transformación completa que va al .sto
        # (Corrección_Yaw * Cuaternión_Sensor * S2S)
        r_final = dict_corrections[seg] * (q_raw_st * s2s_matrices[seg])
        matrix = r_final.as_matrix()

        # 3. Dibujar los ejes del laboratorio como referencia (líneas finas negras)
        ax.quiver(0, 0, 0, 0.8, 0, 0, color='black', alpha=0.3, linewidth=1, linestyle='--') # Lab X
        ax.quiver(0, 0, 0, 0, 0.8, 0, color='black', alpha=0.3, linewidth=1, linestyle='--') # Lab Y
        ax.quiver(0, 0, 0, 0, 0, 0.8, color='black', alpha=0.3, linewidth=1, linestyle='--') # Lab Z

        # 4. Dibujar los ejes del hueso (flechas gruesas)
        for axis_idx in range(3):
            vec = matrix[:, axis_idx]
            ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], 
                      color=colors[axis_idx], linewidth=3, length=1.0)

        ax.set_title(f'SEGMENTO: {seg.upper()}', fontweight='bold')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.view_init(elev=20, azim=45)
        ax.set_axis_off() # Limpiamos para que solo se vean los ejes

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(pca_graphs_folder, 'Galeria_Ejes_Finales.png'), dpi=150)
    plt.close()

def plot_energy_correlation(gyr_data, data_pca, seg_name, pca, pca_graphs_folder):
    """
    Gráfica de correlación que identifica qué eje del sensor corresponde a cada PC.
    """
    # 1. Calculamos intensidades (RMS)
    rms_raw = np.sqrt(np.mean(gyr_data**2, axis=0))
    rms_pca = np.sqrt(np.mean(data_pca**2, axis=0))
    
    # 2. Identificar correspondencia (Mapeo)
    # Buscamos qué eje (X,Y,Z) tiene más peso en cada PC
    axis_names = ['X', 'Y', 'Z']
    mapping = []
    
    if pca is not None:
        for i in range(3):
            # El eje con el valor absoluto más alto en los components es el dominante
            dominant_idx = np.argmax(np.abs(pca.components_[i]))
            mapping.append(axis_names[dominant_idx])
    else:
        # Para la pelvis, el mapeo es directo por cómo construimos la matriz
        mapping = ['X (Fwd)', 'Y (Up)', 'Z (ML)']

    # 3. Configuración del gráfico
    labels = ['SISTEMA SENSOR\n(Original)', 'SISTEMA PCA\n(Calibrado)']
    colors_raw = ['#ff4b4b', '#2ecc71', '#00a8ff'] # Rojo(X), Verde(Y), Azul(Z)
    color_pc1 = '#1F618D' # Azul Oscuro profesional
    color_others = '#AEB6BF' # Gris para secundarios

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # --- BARRA IZQUIERDA: SENSOR ---
    bottom = 0
    for i in range(3):
        val = rms_raw[i]
        ax.bar(labels[0], val, bottom=bottom, color=colors_raw[i], width=0.4)
        if val > 0.3:
            ax.text(labels[0], bottom + val/2, f'Eje {axis_names[i]}\n{val:.2f}', 
                    ha='center', va='center', fontweight='bold', color='white')
        bottom += val

    # --- BARRA DERECHA: PCA ---
    bottom = 0
    pcs = ['PC1', 'PC2', 'PC3']
    for i in range(3):
        val = rms_pca[i]
        color = color_pc1 if i == 0 else color_others
        ax.bar(labels[1], val, bottom=bottom, color=color, width=0.4)
        
        # Etiqueta indicando la procedencia
        source_text = f"{pcs[i]}\n(Viene de {mapping[i]})\n{val:.2f}"
        if val > 0.3:
            ax.text(labels[1], bottom + val/2, source_text, 
                    ha='center', va='center', fontweight='bold', color='white')
        bottom += val

    ax.set_ylabel('Intensidad RMS (rad/s)')
    ax.set_title(f'CORRELACIÓN Y MAPEADO DE EJES: {seg_name}', fontsize=14, fontweight='bold')
    
    # Añadimos una leyenda para aclarar los colores de la izquierda
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#ff4b4b', lw=4, label='Eje X Sensor'),
                       Line2D([0], [0], color='#2ecc71', lw=4, label='Eje Y Sensor'),
                       Line2D([0], [0], color='#00a8ff', lw=4, label='Eje Z Sensor')]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(pca_graphs_folder, f'Energy_Mapping_{seg_name}.png'), dpi=150)
    plt.close()

def plot_gyro_scatter_pca(gyr_data, data_transformed, seg_name, pca, pca_graphs_folder):
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
    plt.savefig(os.path.join(pca_graphs_folder, f'Validacion_{seg_name}.png'), dpi=150)
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

### ------------- BUCLE DE CALIBRACIÓN HÍBRIDO CORREGIDO

def calibrate_segment_functional(df, r_est, r_mov, seg, q_cols, fs): # Añadido fs
    q_static = df.iloc[r_est[0]:r_est[1]][q_cols].dropna().values
    rot_avg  = R.from_quat(q_static).mean()
    v_vert   = rot_avg.inv().apply([0, 0, 1]) 
    v_vert  /= np.linalg.norm(v_vert)

    gyr_cols = get_cols(df, 'Gyr')
    gyr_raw = df.iloc[r_mov[0]:r_mov[1]][gyr_cols].dropna().values
    
    # Uso de fs dinámica para el filtro
    b, a = butter(4, 6.0/(fs/2), btype='low')
    gyr_filtered = filtfilt(b, a, gyr_raw, axis=0)
    
    gyr_projected = gyr_filtered - np.outer(np.dot(gyr_filtered, v_vert), v_vert)

    pca = PCA(n_components=3).fit(gyr_projected)
    v_flex = pca.components_[0]
    v_flex /= np.linalg.norm(v_flex)

    v_fwd = np.cross(v_vert, v_flex)
    v_fwd /= np.linalg.norm(v_fwd)

    mat = np.column_stack((v_fwd, v_vert, v_flex))
    if np.linalg.det(mat) < 0:
        v_flex = -v_flex
        mat = np.column_stack((v_fwd, v_vert, v_flex))

    return R.from_matrix(mat), pca, gyr_raw, v_vert, v_fwd, v_flex

segment_ranges = {}
s2s_matrices = {} 
calib_log = []
q_cols_xsens = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']

print("\n" + "="*50)
print("INICIANDO CALIBRACIÓN FUNCIONAL")
print("="*50)

for seg, df in dict_dfs.items():
    # 1. Segmentar periodos (Estático y Marcha)
    r_est, r_mov, fs_seg = segment_data(df, seg, idx_turn_global)
    fs = fs_seg
    segment_ranges[seg] = (r_est, r_mov)

    if seg == 'pelvis_imu':
        print(f"\n[{seg}] Calibrando por aceleración funcional (Brújula)...")
        # A. Eje Vertical (Y-Up) por Gravedad
        q_static = df.iloc[r_est[0]:r_est[1]][q_cols_xsens].dropna().values
        rot_avg = R.from_quat(q_static).mean()
        v_v = rot_avg.inv().apply([0, 0, 1])
        v_v /= np.linalg.norm(v_v)
        
        # B. Eje ML (Z-Flex) por aceleración funcional
        v_m = get_ml_functional_acc(df, r_mov, v_v) 
        
        # C. Eje Forward (X-Fwd) por producto vectorial (Y x Z = X)
        v_f = np.cross(v_v, v_m)                   
        v_f /= np.linalg.norm(v_f)
        
        # D. Matriz S2S (X, Y, Z)
        mat = np.column_stack((v_f, v_v, v_m))
        det = np.linalg.det(mat)
        if det < 0:
            v_m = -v_m
            mat = np.column_stack((v_f, v_v, v_m))
        
        s2s = R.from_matrix(mat)
        pca_obj = None # La pelvis no genera objeto PCA
        var_pc1_display = "Funcional"
        desv_deg = 90.0
        
        # Datos para visualización
        gyr_cols = get_cols(df, 'Gyr')
        gyr_data = df.iloc[r_mov[0]:r_mov[1]][gyr_cols].dropna().values
        data_plot = s2s.inv().apply(gyr_data)

    else:
        # === MIEMBROS: PCA funcional (Fémures, Tibias, Calcáneos) ===
        # Calibración: v_v=Up, v_f=Forward, v_m=PCA/ML
        s2s, pca_obj, gyr_data, v_v, v_f, v_m = calibrate_segment_functional(df, r_est, r_mov, seg, q_cols_xsens, fs)
        
        # Transformamos los datos al espacio del segmento
        data_plot = pca_obj.transform(gyr_data)
        
        # Variables para Log y Consola
        var_pc1_display = f"{pca_obj.explained_variance_ratio_[0]*100:.1f}%"
        dot_prod = np.dot(v_v, v_m) # Verificamos ortogonalidad con el eje PCA (Z)
        desv_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0)))

        # Evaluación numérica
        rms_raw = np.sqrt(np.mean(gyr_data**2, axis=0)) # RMS en X, Y, Z del sensor
        rms_pca = np.sqrt(np.mean(data_plot**2, axis=0)) # RMS en PC1, PC2, PC3
        
        # Reducción de ruido en el eje secundario (PC2 vs el segundo mejor del sensor)
        reduccion = (1 - (rms_pca[1] / np.sort(rms_raw)[-2])) * 100
        print(f"  > Reducción de crosstalk en eje secundario: {reduccion:.1f}%")

         # =============================================================
        # === AQUÍ LLAMAS A LAS FUNCIONES DE EVALUACIÓN ===
        # =============================================================

        # C. NUEVA LLAMADA: Correlación de energía (Barras apiladas para tu jefe)
        # Asegúrate de que esta línea esté aquí:
        plot_energy_correlation(gyr_data, data_plot, seg, pca_obj, pca_graphs_folder)

    # 3. Guardar Matriz S2S provisional
    s2s_matrices[seg] = s2s
    
    # 4. Log en Consola
    print(f"  > PC1 Var: {var_pc1_display} | Angulo V-Z: {desv_deg:.2f}°")

    # 5. Gráfica Scatter 3D (Común para todos)
    plot_gyro_scatter_pca(gyr_data, data_plot, seg, pca_obj, pca_graphs_folder)

    # 6. Almacenar datos para Excel
    calib_log.append({
        'Segmento': seg,
        'Metodo': 'PCA' if pca_obj else 'Aceleracion',
        'PC1_Variance_%': var_pc1_display,
        'Angulo_V_Z_Deg': desv_deg, 
        'V_Vert_X': v_v[0], 'V_Vert_Y': v_v[1], 'V_Vert_Z': v_v[2],
        'V_Fwd_X': v_f[0], 'V_Fwd_Y': v_f[1], 'V_Fwd_Z': v_f[2],
        'V_ML_X': v_m[0], 'V_ML_Y': v_m[1], 'V_ML_Z': v_m[2] 
    })

s2s_matrices_initial = {seg: R.from_quat(s2s.as_quat()) for seg, s2s in s2s_matrices.items()}

# ==================================================================
# === CORRECCIÓN FÍSICA DEFINITIVA S2S (Alineación con Pelvis) ===
# ==================================================================
print("\n>>> Sincronizando rumbos y realizando FLIP de matrices...")

# 1. Referencia Pelvis
t_p = segment_ranges['pelvis_imu'][0]
q_p = R.from_quat(dict_dfs['pelvis_imu'].iloc[t_p[0]:t_p[1]][q_cols_xsens].dropna().values).mean()
fwd_ref = (q_p * s2s_matrices['pelvis_imu']).apply([1, 0, 0])
fwd_ref[2] = 0; fwd_ref /= np.linalg.norm(fwd_ref)

for seg in s2s_matrices.keys():
    if seg == 'pelvis_imu': continue
    
    t_s = segment_ranges[seg][0]
    q_s = R.from_quat(dict_dfs[seg].iloc[t_s[0]:t_s[1]][q_cols_xsens].dropna().values).mean()
    fwd_seg = (q_s * s2s_matrices[seg]).apply([1, 0, 0])
    fwd_seg[2] = 0; fwd_seg /= np.linalg.norm(fwd_seg)
    
    # Si el hueso apunta hacia atrás respecto a la pelvis, giramos la matriz 180°
    if np.dot(fwd_seg, fwd_ref) < 0:
        flip_180 = R.from_euler('y', 180, degrees=True)
        s2s_matrices[seg] = s2s_matrices[seg] * flip_180
        print(f"  [{seg}] Matriz S2S corregida (Flip 180°)")

# Guardamos el Excel con las matrices ya corregidas
pd.DataFrame(calib_log).to_excel(final_excel, index = False)

for seg, df in dict_dfs.items():
    if seg == 'pelvis_imu': continue
    
    gyr_cols = get_cols(df, 'Gyr')
    gyr_raw = df.iloc[segment_ranges[seg][1][0]:segment_ranges[seg][1][1]][gyr_cols].dropna().values
    
    # Llamamos a la visualización de la corrección
    plot_calibration_evolution(
        gyr_raw, 
        s2s_matrices_initial[seg], 
        s2s_matrices[seg], 
        seg, 
        pca_graphs_folder
    )

# ------------------------------------------------------------------
# Exportación OpenSim
# ------------------------------------------------------------------

### ------------- Archivos .STO

def export_sto(dict_dfs, output_path, segment_ranges, s2s_matrices, fs):
    """
    Exporta datos en formato .sto para OpenSim. 
    Calcula y aplica la corrección de Heading (Yaw) para alinear todos los 
    segmentos con el eje de progresión de la pelvis.
    """
    q_cols = ['Quat_q1', 'Quat_q2', 'Quat_q3', 'Quat_q0']
    is_static = "calibration" in output_path.lower()
    min_len = min([len(df) for df in dict_dfs.values()])

    # 1. CALCULAR CORRECCIONES DE HEADING (Basado siempre en pose estática)
    # Usamos la pelvis estática como referencia de hacia dónde es "adelante"
    t_st_p = segment_ranges['pelvis_imu'][0]
    q_p_st = R.from_quat(dict_dfs['pelvis_imu'].iloc[t_st_p[0]:t_st_p[1]][q_cols].dropna().values).mean()
    q_anat_p = q_p_st * s2s_matrices['pelvis_imu']
    fwd_pelvis = q_anat_p.apply([1, 0, 0])
    fwd_pelvis[2] = 0  # Proyección en plano horizontal
    fwd_pelvis /= np.linalg.norm(fwd_pelvis)

    dict_corrections = {}
    for seg, df in dict_dfs.items():
        t_st = segment_ranges[seg][0]
        q_raw_st = R.from_quat(df.iloc[t_st[0]:t_st[1]][q_cols].dropna().values).mean()
        
        # Orientación anatómica del segmento en el espacio global
        q_anat_world = q_raw_st * s2s_matrices[seg]
        fwd_vec = q_anat_world.apply([1, 0, 0])
        fwd_vec[2] = 0
        fwd_vec /= np.linalg.norm(fwd_vec)

        # Si el Forward del segmento es opuesto a la pelvis, invertimos el rumbo
        if np.dot(fwd_vec, fwd_pelvis) < 0:
            fwd_vec = -fwd_vec

        # Ángulo Yaw para alinear el Forward con el eje X del laboratorio
        yaw_angle = np.arctan2(fwd_vec[1], fwd_vec[0]) + np.pi
        dict_corrections[seg] = R.from_euler('z', -yaw_angle)

    # 2. GENERAR FILAS DE DATOS
    data_rows = []
    frames = 1 if is_static else min_len

    for i in range(frames):
        row = [i / fs]
        for seg, df in dict_dfs.items():
            if is_static:
                # Promediamos la pose estática para el archivo de calibración
                t = segment_ranges[seg][0]
                q_raw = R.from_quat(df.iloc[t[0]:t[1]][q_cols].dropna().values).mean()
            else:
                # Orientación instantánea para el archivo de movimiento
                q_raw = R.from_quat(df.iloc[i][q_cols].values)

            # Aplicamos la cadena de transformación: Corrección_Yaw * (Sensor * S2S)
            r_f = dict_corrections[seg] * (q_raw * s2s_matrices[seg])
            q_f = r_f.as_quat()
            
            # Convención de cuaternión: escalar (w) siempre positivo
            if q_f[3] < 0: q_f = -q_f
            row.append(f"{q_f[3]},{q_f[0]},{q_f[1]},{q_f[2]}")
            
        data_rows.append(row)

    # 3. ESCRITURA FÍSICA DEL ARCHIVO
    with open(output_path, 'w') as f:
        f.write(f"DataRate={fs:.6f}\nDataType=Quaternion\nversion=3\n"
                "OpenSimVersion=4.5\nendheader\n")
        
        pd.DataFrame(
            data_rows,
            columns=['time'] + list(dict_dfs.keys())
        ).to_csv(f, sep='\t', index=False)

    tipo = "ESTÁTICO" if is_static else "MOVIMIENTO"
    print(f"Exportado: {os.path.basename(output_path)} ({tipo}) a {fs:.1f} Hz")

    return dict_corrections   

# Guardamos el diccionario que devuelve la función en una variable
dict_corrections = export_sto(dict_dfs, sto_movement, segment_ranges, s2s_matrices, fs)

# Para el archivo de calibración no hace falta guardarlo otra vez (es el mismo)
export_sto(dict_dfs, sto_calibration, segment_ranges, s2s_matrices, fs)

### ------------- Archivo XML
root = ET.Element("OpenSimDocument", Version = "40000")
placer = ET.SubElement(root, "IMUPlacer")
ET.SubElement(placer, "base_imu_label").text = 'pelvis_imu'
ET.SubElement(placer, "base_heading_axis").text = '-x'
ET.SubElement(placer, "sensor_to_opensim_rotations").text = "-1.5707963267948966 0 0"
ET.SubElement(placer, "orientation_file_for_calibration").text = os.path.basename(sto_calibration)
with open(xml_placer, 'w') as f: f.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="   "))

plot_axes_gallery(dict_dfs, s2s_matrices, segment_ranges, dict_corrections, pca_graphs_folder)
plot_sensor_orientation_in_segment(s2s_matrices, pca_graphs_folder)

print(f"\n>>> PROCESO FINALIZADO CON ÉXITO")