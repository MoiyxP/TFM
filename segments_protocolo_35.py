"""
segments_protocolo_35.py
==========================
Definición de los 7 segmentos corporales (Pelvis, Muslo, Pierna, Pie -
bilateral) para el protocolo de 35 marcadores verificado en
Running_trial_3.c3d / Static_trial_1.c3d (L_HEAD, R_HEAD, SGL, ...,
L_FLE, L_TTC, L_FAL, L_FCC, L_FM2, etc.).

Extraído tal cual de la definición que antes vivía embebida en main.py
(sin cambios de lógica), para que main.py pueda elegir entre este
protocolo y otros (ver segments_protocolo_94.py, detectar_protocolo.py)
sin tener las definiciones hardcodeadas en un solo archivo.

IMPORTANTE - orientación verificada en Running_trial_3.c3d:
    X (lab) -> antero-posterior | Y (lab) -> +Y hacia la IZQUIERDA del
    sujeto | Z (lab) -> vertical. Si cambia el laboratorio, re-verificar
    con la visualización 3D antes de confiar en los signos.
"""

from typing import Dict, NamedTuple, Tuple


class SegmentDef(NamedTuple):
    origin: Tuple[str, ...]
    axis_1: Tuple[str, str]
    axis_2: Tuple[str, str]
    axes_name: str
    axis_to_recalculate: str


SEGMENTS: Dict[str, SegmentDef] = {
    "PV": SegmentDef(
        origin=("L_IAS", "R_IAS", "SACR"),
        axis_1=("SACR", "L_IAS"),
        axis_2=("L_IAS", "R_IAS"),
        axes_name="xz",
        axis_to_recalculate="x",
    ),
    "TH_R": SegmentDef(
        origin=("R_FLE",),
        axis_1=("R_PAS", "R_IAS"),
        axis_2=("R_FLE", "R_IAS"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "TH_L": SegmentDef(
        origin=("L_FLE",),
        axis_1=("L_PAS", "L_IAS"),
        axis_2=("L_FLE", "L_IAS"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "SH_R": SegmentDef(
        origin=("R_FAL",),
        axis_1=("R_FAL", "R_TTC"),
        axis_2=("R_FAL", "R_FLE"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "SH_L": SegmentDef(
        origin=("L_FAL",),
        axis_1=("L_FAL", "L_TTC"),
        axis_2=("L_FAL", "L_FLE"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "FT_R": SegmentDef(
        origin=("R_FCC",),
        axis_1=("R_FCC", "R_FAL"),
        axis_2=("R_FCC", "R_FM2"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "FT_L": SegmentDef(
        origin=("L_FCC",),
        axis_1=("L_FCC", "L_FAL"),
        axis_2=("L_FCC", "L_FM2"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
}

XSENS_NAME_MAP = {
    "PV": "PV", "TH_R": "TH DER", "TH_L": "TH IZQ",
    "SH_R": "SH DER", "SH_L": "SH IZQ", "FT_R": "FT DER", "FT_L": "FT IZQ",
}

MARCADORES_REQUERIDOS = sorted({
    m for seg in SEGMENTS.values()
    for m in (*seg.origin, *seg.axis_1, *seg.axis_2)
})
