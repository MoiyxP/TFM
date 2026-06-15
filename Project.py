import pandas as pd
import pyomeca
import os
import numpy as np
import json
import matplotlib as plt
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

# ----------------------------------
# Aplicación del código.
# ----------------------------------

# Evaluación de la señal c3d
markers, analogs = read_c3d(movement_c3d)
report = gap_report(markers)    

with open("results.json", "w") as f:
    json.dump(report, f, indent=4)           

# Importación de los datos de csv
metadata, quats, euler, euavg = import_excel_data(ruta_excel)
