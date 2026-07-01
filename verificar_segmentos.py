"""
verificar_segmentos.py
========================
Corre, para los 7 segmentos definidos en segments_protocolo_94.py, los 3
chequeos que fuimos usando "a mano" a lo largo de la sesión, de una sola
vez, con un reporte claro de qué pasa y qué no. Pensado para correr cada
vez que cambies algo en la definición de segmentos, ANTES de mirar el
visualizador o confiar en resultados dinámicos.

USO:
    python verificar_segmentos.py

QUÉ CHEQUEA (en orden de importancia):
    1. ORTONORMALIDAD: determinante=1 y ejes perpendiculares entre sí, en
       el trial dinámico. Si esto falla, hay un bug real de construcción
       geométrica (ver el caso que encontramos con axis_to_recalculate).

    2. DIRECCIÓN EN EL ESTÁTICO: cada eje (Y=vertical, X=anterior,
       Z=lateral) debe tener su componente dominante en la dirección
       anatómica esperada, CON EL MISMO SIGNO en ambos lados (R y L).
       Si un lado sale con signo opuesto al otro para el mismo eje, es
       la señal de que hay un problema de simetría espejo (ver el caso
       de TH_L/SH_L/FT_L en la sesión).

    3. ESTABILIDAD EN EL ESTÁTICO: si el sujeto estuvo quieto durante la
       N-pose, la desviación estándar de los ángulos de Euler debería
       ser muy baja (<1-2°, típicamente). Si sale alta, o el sujeto se
       movió, o hay inestabilidad numérica en el cálculo para ese
       segmento en particular.

Este script NO reemplaza mirar el visualizador - lo complementa. El
visualizador te deja VER si algo se ve razonable; este script te da
NÚMEROS concretos para confirmar (o descartar) esa sospecha.
"""

import sys
import numpy as np

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

RUTA_MAIN = r"C:\Users\DanielIordanov\Desktop\VictorMV-biomech\TFM\Code\TFM\Project.py"
C3D_ESTATICO = r"C:\Users\DanielIordanov\Desktop\VictorMV-biomech\TFM\Code\TFM\Archivos\Tutorial modelos\Visual3D_Tutorial1_SampleC3DFiles\Sample_C3D_Files\Sub01_StandingStaticCal01.c3d"
C3D_DINAMICO = r"C:\Users\DanielIordanov\Desktop\VictorMV-biomech\TFM\Code\TFM\Archivos\Tutorial modelos\Visual3D_Tutorial1_SampleC3DFiles\Sample_C3D_Files\Sub01_Walk001.c3d"

# Umbrales de referencia (ajustables si tu criterio es distinto)
UMBRAL_DET_TOLERANCIA = 1e-4       # |det - 1| debe ser menor a esto
UMBRAL_ORTOGONALIDAD = 1e-6        # productos punto entre ejes deben ser menores a esto
UMBRAL_STD_ESTATICO_DEG = 2.0      # desviación estándar de Euler en N-pose, en grados


def main_verificacion():
    sys.path.insert(0, RUTA_MAIN)
    import Project as main
    import segments_protocolo_94 as seg94

    main.SEGMENTS = seg94.SEGMENTS

    print("Leyendo archivos...")
    markers_est, analogs_est = main.read_c3d(C3D_ESTATICO)
    int_est, _ = main.interpolate_c3d(markers_est, analogs_est, "akima")

    markers_din, analogs_din = main.read_c3d(C3D_DINAMICO)
    int_din, _ = main.interpolate_c3d(markers_din, analogs_din, "akima")

    print("\n" + "=" * 78)
    print("CHEQUEO 1 - ORTONORMALIDAD (en el trial DINÁMICO, todos los frames)")
    print("=" * 78)
    todo_ok_1 = True
    for seg_name, seg_def in seg94.SEGMENTS.items():
        R = main.construir_matriz_rotacion(int_din, seg_def)
        n_frames = R.shape[2]
        dets = np.array([np.linalg.det(R[:, :, f]) for f in range(n_frames)])
        max_desvio_det = np.max(np.abs(dets - 1))

        x, y, z = R[:, 0, :], R[:, 1, :], R[:, 2, :]
        max_dot_xy = np.max(np.abs(np.sum(x * y, axis=0)))
        max_dot_yz = np.max(np.abs(np.sum(y * z, axis=0)))
        max_dot_xz = np.max(np.abs(np.sum(x * z, axis=0)))
        max_dot = max(max_dot_xy, max_dot_yz, max_dot_xz)

        ok = max_desvio_det < UMBRAL_DET_TOLERANCIA and max_dot < UMBRAL_ORTOGONALIDAD
        todo_ok_1 &= ok
        estado = "OK" if ok else "*** FALLA ***"
        print(f"  {seg_name:6s}: max|det-1|={max_desvio_det:.2e}  max|producto punto|={max_dot:.2e}  [{estado}]")

    print("\n" + "=" * 78)
    print("CHEQUEO 2 - DIRECCIÓN EN EL ESTÁTICO (componente dominante de cada eje)")
    print("=" * 78)
    print("Referencia: Y->vertical(Z_lab), X->anterior(*), Z->lateral(*)")
    print("(* la componente 'anterior'/'lateral' del laboratorio se infiere de PV, no asumida)")

    # Inferir qué eje del laboratorio es "anterior" mirando el eje X de PV
    # (que no tiene ambigüedad de lado R/L, es un buen ancla)
    R_pv = main.construir_matriz_rotacion(int_est, seg94.SEGMENTS["PV"])
    x_pv = R_pv[:, 0, 0]
    eje_anterior_lab = int(np.argmax(np.abs(x_pv)))  # 0=X_lab, 1=Y_lab, 2=Z_lab
    nombres_lab = ["X_lab", "Y_lab", "Z_lab"]
    print(f"(eje del laboratorio inferido como 'anterior': {nombres_lab[eje_anterior_lab]}, desde PV)\n")

    signos_por_lado = {}  # {(base_seg, eje_local): {"R": signo, "L": signo}}
    todo_ok_2 = True
    for seg_name, seg_def in seg94.SEGMENTS.items():
        R = main.construir_matriz_rotacion(int_est, seg_def)
        y0 = R[:, 1, 0]
        x0 = R[:, 0, 0]

        dominante_y = int(np.argmax(np.abs(y0)))
        signo_y_vertical = "OK (vertical, arriba)" if (dominante_y == 2 and y0[2] > 0) else f"*** {nombres_lab[dominante_y]}={y0[dominante_y]:+.2f} ***"

        dominante_x = int(np.argmax(np.abs(x0)))
        signo_x_anterior = "OK (anterior)" if dominante_x == eje_anterior_lab else f"*** dominante={nombres_lab[dominante_x]} (no anterior) ***"

        print(f"  {seg_name:6s}: eje Y={y0.round(2)}  [{signo_y_vertical}]")
        print(f"          eje X={x0.round(2)}  [{signo_x_anterior}]")

        if "***" in signo_y_vertical or "***" in signo_x_anterior:
            todo_ok_2 = False

        # Guardar signo del eje anterior para comparar L vs R del mismo segmento base
        base = seg_name.rsplit("_", 1)[0] if seg_name.endswith(("_R", "_L")) else None
        lado = seg_name[-1] if seg_name.endswith(("_R", "_L")) else None
        if base:
            signos_por_lado.setdefault(base, {})[lado] = np.sign(x0[eje_anterior_lab])

    print("\n  --- Comparación de signo del eje anterior entre lados R/L (mismo segmento) ---")
    for base, signos in signos_por_lado.items():
        if "R" in signos and "L" in signos:
            coincide = signos["R"] == signos["L"]
            todo_ok_2 &= coincide
            estado = "OK, mismo signo" if coincide else "*** SIGNO OPUESTO ENTRE LADOS ***"
            print(f"  {base}: R={signos['R']:+.0f}  L={signos['L']:+.0f}  [{estado}]")

    print("\n" + "=" * 78)
    print(f"CHEQUEO 3 - ESTABILIDAD EN EL ESTÁTICO (std de Euler, umbral {UMBRAL_STD_ESTATICO_DEG}°)")
    print("=" * 78)
    todo_ok_3 = True
    for seg_name, seg_def in seg94.SEGMENTS.items():
        R = main.construir_matriz_rotacion(int_est, seg_def)
        df_euler = main.matriz_a_euler_zxy(R)
        std_max = df_euler[["Euler_Z", "Euler_X", "Euler_Y"]].std().max()
        ok = std_max < UMBRAL_STD_ESTATICO_DEG
        todo_ok_3 &= ok
        estado = "OK" if ok else "*** ALTA VARIACIÓN ***"
        print(f"  {seg_name:6s}: std máxima = {std_max:.2f}°  [{estado}]")

    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(f"  1. Ortonormalidad:        {'TODO OK' if todo_ok_1 else 'HAY FALLAS - revisar arriba'}")
    print(f"  2. Dirección/simetría:    {'TODO OK' if todo_ok_2 else 'HAY FALLAS - revisar arriba'}")
    print(f"  3. Estabilidad estático:  {'TODO OK' if todo_ok_3 else 'HAY FALLAS - revisar arriba'}")


if __name__ == "__main__":
    main_verificacion()
