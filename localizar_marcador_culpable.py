"""
localizar_marcador_culpable.py
=================================
Diagnóstico dirigido para el caso ya confirmado: el segmento PV tiene un
salto de orientación de ~140-150° en exactamente 2 frames (282 y 353,
t≈1.410s y t≈1.765s), mientras que TH_R, SH_R, etc. están limpios.

Como PV se construye solo con 3 marcadores (LIAS, RIAS, SCRM - ver
segments_protocolo_94.py), este script imprime la posición de esos 3
marcadores en una ventana alrededor de cada frame sospechoso, para ver
a ojo cuál de los tres "salta" de posición de forma anómala y vuelve
inmediatamente al frame siguiente (la firma típica de un mislabeling
puntual de marcador en captura óptica).

Uso (con tus datos reales, después de leer e interpolar el c3d de
movimiento igual que en main.py):

    
"""
import numpy as np


def inspeccionar_marcadores_pv(markers, frames_sospechosos, ventana=3,
                                 nombres_marcadores=("LIAS", "RIAS", "SCRM")):
    """
    Imprime, para cada marcador de PV, su posición (x,y,z) en una ventana
    de +/- `ventana` frames alrededor de cada frame sospechoso, y el
    desplazamiento respecto al frame anterior - el marcador culpable
    mostrará un salto grande SOLO en ese frame, volviendo a la normalidad
    inmediatamente después (a diferencia de un movimiento real, que sería
    gradual en los frames vecinos).
    """
    time = markers.time.values

    for frame_central in frames_sospechosos:
        print(f"\n{'='*70}")
        print(f"Frame sospechoso: {frame_central}  (t={time[frame_central]:.4f}s)")
        print(f"{'='*70}")

        rango = range(max(0, frame_central - ventana), min(len(time), frame_central + ventana + 1))

        for nombre in nombres_marcadores:
            pos = markers.sel(channel=[nombre]).values[:3, 0, :]  # (3, n_frames)
            print(f"\n  Marcador {nombre}:")
            prev = None
            for i in rango:
                x, y, z = pos[:, i]
                marca = ""
                if prev is not None:
                    salto = np.linalg.norm(pos[:, i] - prev)
                    if salto > 15:  # mm, ajustar si tu escala/frecuencia es distinta
                        marca = f"   <-- SALTO {salto:.1f}mm respecto al frame anterior"
                print(f"    frame {i:>4d}  t={time[i]:.4f}  pos=({x:8.2f}, {y:8.2f}, {z:8.2f}){marca}")
                prev = pos[:, i]


def resumen_culpable(markers, frames_sospechosos, nombres_marcadores=("LIAS", "RIAS", "SCRM"), umbral_mm=15.0):
    """
    Versión resumida: para cada frame sospechoso, indica directamente
    cuál(es) marcador(es) tuvieron un salto > umbral_mm justo en ese
    frame (comparado con el frame anterior).
    """
    time = markers.time.values
    print(f"{'Frame':>6s}  {'t(s)':>8s}  {'Marcador culpable':<15s}  {'Salto (mm)':>10s}")
    print("-" * 50)
    for frame in frames_sospechosos:
        if frame == 0:
            continue
        for nombre in nombres_marcadores:
            pos = markers.sel(channel=[nombre]).values[:3, 0, :]
            salto = np.linalg.norm(pos[:, frame] - pos[:, frame - 1])
            marca = " <-- CULPABLE" if salto > umbral_mm else ""
            print(f"{frame:>6d}  {time[frame]:>8.4f}  {nombre:<15s}  {salto:>10.1f}{marca}")

