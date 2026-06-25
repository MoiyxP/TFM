"""
segments_protocolo_94.py
==========================
Definición de los 7 segmentos corporales (Pelvis, Muslo, Pierna, Pie -
bilateral) para el protocolo de 94 marcadores verificado en
Sub01_StandingStaticCal01.c3d (CTRHD, LARM1-4, LSHK1-4, LTH1-4, etc.).

Construido con la MISMA estructura que segments.py (el protocolo de 35
marcadores: L_HEAD, L_SAE, etc.), pero con los nombres de marcador de
este protocolo distinto. La lógica de construir_matriz_rotacion,
calcular_orientaciones_desde_markers, calibración con N-pose, etc. en
main.py NO depende de los nombres de marcador - son genéricas. Solo
este archivo cambia entre protocolos.

IMPORTANTE - orientación verificada en Sub01_StandingStaticCal01.c3d:
    Z (lab) -> vertical (cabeza ~1800mm, pie ~20-90mm)
    Mismo patrón que el protocolo de 35 marcadores para X/Y, pero NO
    verificado explícitamente con un trial dinámico de este protocolo -
    revisar con la visualización 3D antes de confiar en los signos si
    los resultados angulares salen raros.

Marcadores elegidos por segmento, verificados con distancias geométricas
reales (ver historial de la sesión) contra suposiciones por nombre:
    PV:   LIAS, RIAS (espina ilíaca antero-superior), SCRM (sacro)
    TH:   FLE (epicóndilo lateral fémur, rodilla), IAS/IPS (cadera)
    SH:   FAL (maléolo lateral), TTC (tuberosidad tibial), FLE (rodilla)
    FT:   FCC (calcáneo), FAL (maléolo lateral), TOE (dedo)

Marcadores NO usados por ambigüedad sin resolver del todo (clusters
técnicos LARM/LTH/LSHK 1-4, ULB/LLB, FAX, FT1-3): no son necesarios
para esta definición de 7 segmentos con 3 marcadores cada uno, y usar
clusters técnicos en vez de landmarks anatómicos requeriría una lógica
de construcción de eje distinta (a partir de un cluster rígido, no de
2 vectores anatómicos) que no está implementada aquí.
"""

from typing import Dict, NamedTuple, Tuple


class SegmentDef(NamedTuple):
    origin: Tuple[str, ...]
    axis_1: Tuple[str, str]
    axis_2: Tuple[str, str]
    axes_name: str
    axis_to_recalculate: str


SEGMENTS: Dict[str, SegmentDef] = {
    # PELVIS
    "PV": SegmentDef(
        origin=("LIAS", "RIAS", "SCRM"),
        axis_1=("SCRM", "LIAS"),
        axis_2=("LIAS", "RIAS"),
        axes_name="xz",
        axis_to_recalculate="x",
    ),
    # MUSLO (Thigh)
    "TH_R": SegmentDef(
        origin=("RFLE",),
        axis_1=("RIPS", "RIAS"),
        axis_2=("RFLE", "RIAS"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "TH_L": SegmentDef(
        origin=("LFLE",),
        axis_1=("LIPS", "LIAS"),
        axis_2=("LFLE", "LIAS"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    # PIERNA (Shank)
    "SH_R": SegmentDef(
        origin=("RFAL",),
        axis_1=("RFAL", "RTTC"),
        axis_2=("RFAL", "RFLE"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "SH_L": SegmentDef(
        origin=("LFAL",),
        axis_1=("LFAL", "LTTC"),
        axis_2=("LFAL", "LFLE"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    # PIE (Foot)
    "FT_R": SegmentDef(
        origin=("RFCC",),
        axis_1=("RFCC", "RFAL"),
        axis_2=("RFCC", "RTOE"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
    "FT_L": SegmentDef(
        origin=("LFCC",),
        axis_1=("LFCC", "LFAL"),
        axis_2=("LFCC", "LTOE"),
        axes_name="xy",
        axis_to_recalculate="x",
    ),
}

XSENS_NAME_MAP = {
    "PV": "PV", "TH_R": "TH DER", "TH_L": "TH IZQ",
    "SH_R": "SH DER", "SH_L": "SH IZQ", "FT_R": "FT DER", "FT_L": "FT IZQ",
}

# Lista de todos los marcadores que esta definición necesita - usada para
# verificar al arrancar que el c3d los tiene todos (ver verificar_marcadores
# en main.py).
MARCADORES_REQUERIDOS = sorted({
    m for seg in SEGMENTS.values()
    for m in (*seg.origin, *seg.axis_1, *seg.axis_2)
})
