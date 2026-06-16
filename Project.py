import pandas as pd
import pyomeca
import os
import numpy as np
import xarray
import xlsxwriter
import json
import matplotlib.pyplot as plt
import ezc3d as ez
from pyomeca import Angles, Markers, Analogs

ruta_excel = r"C:\Users\DanielIordanov\Desktop\VictorMV-biomech\TFM\Code\TFM\Archivos\prueba.xlsx"
static_c3d = r"C:\Users\DanielIordanov\Desktop\VictorMV-biomech\TFM\Code\TFM\Archivos\Static trial 1.c3d"
movement_c3d = r"C:\Users\DanielIordanov\Desktop\VictorMV-biomech\TFM\Code\TFM\Archivos\Running trial 3.c3d"

# ----------------------------------
# Lógica del código.
# ----------------------------------

def read_c3d(ruta):
    """
    Lee un archivo .c3d y extrae los marcadores y, si existen, las señales
    analógicas.

    """
    markers = Markers.from_c3d(ruta)
    df_markers = markers.meca.to_wide_dataframe()
    df_markers.to_csv("markers_c3d.csv", sep=';', decimal=',', encoding='utf-8-sig')
 
    try:
        analogs = Analogs.from_c3d(ruta)
    except Exception:
        print("No analogs in this file")
        analogs = None

    if analogs:
        df_analogs = analogs.meca.to_wide_dataframe()
        df_analogs.to_csv("manalogs_c3d.csv", sep=';', decimal=',', encoding='utf-8-sig')
 
    return markers, analogs

def import_excel_data(ruta):
    """
    Lectura de los excel de datos
    """
    df_metadata = pd.read_excel(ruta, sheet_name=0)
    df_quats = pd.read_excel(ruta, sheet_name=1)
    df_euler = pd.read_excel(ruta, sheet_name=2)
    df_euavg = pd.read_excel(ruta, sheet_name=3) 

    # Establecemos el tiempo como índice para facilitar comparaciones temporales
    df_quats= df_quats.set_index('Tiempo(s)')
    df_euler = df_euler.set_index('Tiempo(s)')
    df_euavg = df_euavg.set_index('Tiempo(s)')
    
    return df_metadata, df_quats, df_euler, df_euavg

def evaluate_marker_gaps(markers, marker_name, thresholds=(10, 50)):
    """
    Evalúa la calidad de un marcador 3D detectando y clasificando gaps.

    """
 
    small_threshold, medium_threshold = thresholds
 
    # ------ Máscara de NaNs
    
    data = markers.sel(channel=marker_name).values
    mask = np.isnan(data).any(axis=0)  # True donde el frame tiene algún NaN
 
    n_frames = len(mask)
 
    # ------ Pérdida global

    loss_fraction = float(mask.mean())
 
    # ------ Detectar gaps (transiciones)

    changes = np.diff(mask.astype(int))
 
    gap_starts = np.where(changes == 1)[0] + 1
    gap_ends = np.where(changes == -1)[0] + 1
 
    # bordes: gap al inicio o al final del trial
    if mask[0]:
        gap_starts = np.r_[0, gap_starts]
    if mask[-1]:
        gap_ends = np.r_[gap_ends, n_frames]
 
    # ------ Caso sin gaps 

    if len(gap_starts) == 0:
        return {
            "marker": marker_name,
            "loss_fraction": loss_fraction,
            "n_gaps": 0,
            "gap_lengths": [],
            "gap_details": []
        }
 
    # ------ Longitudes de gaps

    gap_lengths = gap_ends - gap_starts
 
    # ------ Análisis por gap

    gap_details = []
 
    for i, (s, e, length) in enumerate(zip(gap_starts, gap_ends, gap_lengths)):
 
        fraction = length / n_frames
 
        if length < small_threshold:
            severity = "small"
        elif length < medium_threshold:
            severity = "medium"
        else:
            severity = "large"
 
        gap_details.append({
            "gap_id": i,
            "start": int(s),
            "end": int(e),
            "length_frames": int(length),
            "fraction_of_trial": float(fraction),
            "severity": severity
        })
 
    # ------ Resultado final
    return {
        "marker": marker_name,
        "loss_fraction": loss_fraction,
        "n_gaps": len(gap_lengths),
        "gap_lengths": gap_lengths.tolist(),
        "gap_details": gap_details
    }
 
def gap_report(markers, thresholds=(10, 50)):
    """
    Genera un informe de gaps para todos los marcadores de un trial.
 
    """
    marker_names = markers.channel.values.tolist()
 
    return {
        name: evaluate_marker_gaps(markers, name, thresholds=thresholds)
        for name in marker_names
    }

def interpolate_c3d(markers, analogs, method):
    """
    Vamos a interpolar los datos de los marcadores
    independientemente de la longitud de sus huecos

    """
    # Hacemos la interpolación de los datos de los marcadores 
    int_markers = markers.interpolate_na(dim="time", method=method)

    # Rellenamos los posibles huecos al final de la señal
    int_markers = int_markers.ffill(dim="time")

    # Rellenamos los posibles huecos al principio de la señal
    int_markers = int_markers.bfill(dim="time")

    # Pasamos primero a un df para poder pasarlos a un excel ordenado
    df_markers = int_markers.meca.to_wide_dataframe()
    df_markers.to_csv("int_markers_c3d.csv", sep=';', decimal=',', encoding='utf-8-sig')

    if analogs is not None and analogs.size > 0:
        try:
            int_analogs = analogs.interpolate_na(dim="time", method=method)
            int_analogs = int_analogs.ffill(dim="time").bfill(dim="time")

            # Exportar analógicos solo si existen
            df_analogs = int_analogs.meca.to_wide_dataframe()
            df_analogs.to_csv("int_analogs_c3d.csv", sep=';', decimal=',', encoding='utf-8-sig')
        except Exception as e:
            print(f"Error al procesar analógicos: {e}")
    else:
        print("No se detectaron datos analógicos en este archivo.")
        int_analogs = 'None'

    return int_markers, int_analogs

def save_global_report(ruta_salida, metadata, quats, euler, euavg, int_markers, int_analogs, gap_report_dict):
    """
    Crea un único archivo Excel con toda la información procesada y ajuste automático de columnas.
    """
    with pd.ExcelWriter(ruta_salida, engine='xlsxwriter') as writer:
    
        def autofit(df, sheet_name, include_index=True):
            """
            Ajusta el tamaño de las columnas del excel
            
            """
            worksheet = writer.sheets[sheet_name]
            # Si incluimos el índice (Tiempo), lo tratamos como la columna 0
            offset = 1 if include_index else 0
            
            if include_index:
                # Ajustar ancho del índice
                idx_name = str(df.index.name) if df.index.name else ""
                max_len = max([len(idx_name)] + [len(str(x)) for x in df.index[:100]]) + 2
                worksheet.set_column(0, 0, max_len)

            # Ajustar ancho de las columnas de datos
            for i, col in enumerate(df.columns):
                # Comparamos largo del nombre de columna vs los datos (primeras 100 filas por eficiencia)
                max_len = max([len(str(col))] + [len(str(x)) for x in df[col][:100]]) + 2
                worksheet.set_column(i + offset, i + offset, max_len)

        # Guardamos los datos de Xsens
        metadata.to_excel(writer, sheet_name='Xsens_Metadata', index=False)
        autofit(metadata, 'Xsens_Metadata', include_index=False)

        quats.to_excel(writer, sheet_name='Xsens_Quats')
        autofit(quats, 'Xsens_Quats')

        euler.to_excel(writer, sheet_name='Xsens_Euler')
        autofit(euler, 'Xsens_Euler')

        euavg.to_excel(writer, sheet_name='Xsens_Euler_Avg')
        autofit(euavg, 'Xsens_Euler_Avg')

        # Guardamos los datos del c3d ya interpolados
        df_markers = int_markers.meca.to_wide_dataframe()
        df_markers.to_excel(writer, sheet_name='C3D_Markers')
        autofit(df_markers, 'C3D_Markers')

        if int_analogs is not None and not isinstance(int_analogs, str):
            df_analogs = int_analogs.meca.to_wide_dataframe()
            df_analogs.to_excel(writer, sheet_name='C3D_Analogs')
            autofit(df_analogs, 'C3D_Analogs')

        # Convertimos el informe de gaps a una tabla de Excel
        summary_data = []
        for marker, info in gap_report_dict.items():
            summary_data.append({
                "Marcador": marker,
                "Pérdida_Total_%": round(info['loss_fraction'] * 100, 2),
                "Num_Gaps": info['n_gaps'],
                "Max_Gap_Frames": max(info['gap_lengths']) if info['gap_lengths'] else 0
            })
        
        df_gap_summary = pd.DataFrame(summary_data)
        sheet_gaps = 'C3D Marker - Gaps Analysis'
        df_gap_summary.to_excel(writer, sheet_name=sheet_gaps, index=False)
        autofit(df_gap_summary, sheet_gaps, include_index=False)

def plot_gap_map(report_dict):
    """
    Crea un mapa de calor que muestra dónde faltan datos para cada marcador.
    """
    # Extraer los estados de los gaps de forma visual
    marker_names = list(report_dict.keys())
    # Suponemos que todos tienen la misma longitud de frames
    n_frames = report_dict[marker_names[0]]['n_gaps'] # solo para referencia
    
    plt.figure(figsize=(12, 8))
    
    for i, name in enumerate(marker_names):
        details = report_dict[name]['gap_details']
        for gap in details:
            color = 'red' if gap['severity'] == 'large' else 'orange' if gap['severity'] == 'medium' else 'yellow'
            plt.hlines(i, gap['start'], gap['end'], colors=color, lw=6)
            
    plt.yticks(range(len(marker_names)), marker_names)
    plt.title("Mapa de pérdida de marcadores (Rojo=Crítico, Amarillo=Leve)")
    plt.xlabel("Frames")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ----------------------------------
# Aplicación del código.
# ----------------------------------

# Evaluación de calidad de la señal c3d
markers, analogs = read_c3d(movement_c3d)
report = gap_report(markers)    
with open("results.json", "w") as f: # Lo visualizamos en un json
    json.dump(report, f, indent=4)           

# Vamos a interpolar todos los gaps, independientemente de su tamaño
int_markers, int_analogs = interpolate_c3d(markers, analogs, 'akima')

# Importación de los datos de csv
metadata, quats, euler, euavg = import_excel_data(ruta_excel)

# Visualización de datos
plot_gap_map(report)

# Guardamos todo en un excel global
ruta_final = "Informe_final.xlsx"
save_global_report(ruta_final, metadata, quats, euler, euavg, int_markers, int_analogs, report)
