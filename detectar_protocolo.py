"""
detectar_protocolo.py
=======================
Detecta automáticamente qué protocolo de marcadores corresponde a un
archivo .c3d, comparando sus labels contra los MARCADORES_REQUERIDOS de
cada módulo segments_protocolo_*.py conocido, y devuelve el módulo
SEGMENTS/XSENS_NAME_MAP correspondiente para que main.py lo use sin
necesidad de elegirlo a mano.

Agregar un protocolo nuevo:
    1. Crear segments_protocolo_NOMBRE.py con SEGMENTS, XSENS_NAME_MAP,
       y MARCADORES_REQUERIDOS (ver segments_protocolo_35.py o
       segments_protocolo_94.py como plantilla).
    2. Agregar el nombre del módulo a PROTOCOLOS_CONOCIDOS más abajo.

NOTA: este módulo agrega su propio directorio a sys.path (ver más abajo)
para que los segments_protocolo_*.py se puedan importar sin importar
desde qué cwd se esté ejecutando main.py (main() hace os.chdir a la
carpeta de salida antes de llamar a la detección).
"""

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Optional

# Asegurar que el directorio donde vive este archivo esté en sys.path,
# para que los imports de segments_protocolo_* funcionen sin importar
# el cwd actual del proceso (ver nota arriba).
_DIR_DE_ESTE_ARCHIVO = os.path.dirname(os.path.abspath(__file__))
if _DIR_DE_ESTE_ARCHIVO not in sys.path:
    sys.path.insert(0, _DIR_DE_ESTE_ARCHIVO)

PROTOCOLOS_CONOCIDOS = [
    "segments_protocolo_35",
    "segments_protocolo_94",
]


@dataclass
class ProtocoloDetectado:
    nombre_modulo: str
    marcadores_faltantes: list
    coincidencia_exacta: bool  # True si el c3d tiene EXACTAMENTE los marcadores requeridos (no de más/menos)


def buscar_protocolo(labels_c3d) -> Optional[ProtocoloDetectado]:
    """
    Compara los labels de un c3d contra MARCADORES_REQUERIDOS de cada
    protocolo conocido. Devuelve el primer protocolo donde TODOS los
    marcadores requeridos están presentes (coincidencia_exacta indica si
    además el set de labels es idéntico, útil para distinguir variantes
    del mismo protocolo base). Si ninguno calza, devuelve None.
    """
    labels_set = set(labels_c3d)

    candidatos = []
    for nombre_modulo in PROTOCOLOS_CONOCIDOS:
        modulo = importlib.import_module(nombre_modulo)
        requeridos = set(modulo.MARCADORES_REQUERIDOS)
        faltantes = requeridos - labels_set

        if not faltantes:
            candidatos.append(ProtocoloDetectado(
                nombre_modulo=nombre_modulo,
                marcadores_faltantes=[],
                coincidencia_exacta=(requeridos == labels_set),
            ))

    if not candidatos:
        return None

    # Si hay más de un candidato que calza (subconjuntos compatibles),
    # preferir el de coincidencia exacta; si ninguno es exacto, el primero.
    exactos = [c for c in candidatos if c.coincidencia_exacta]
    return exactos[0] if exactos else candidatos[0]


def cargar_segments(nombre_modulo: str):
    """Importa un módulo segments_protocolo_* y devuelve (SEGMENTS, XSENS_NAME_MAP)."""
    modulo = importlib.import_module(nombre_modulo)
    return modulo.SEGMENTS, modulo.XSENS_NAME_MAP


def detectar_y_cargar(labels_c3d, verbose: bool = True):
    """
    Detecta el protocolo y devuelve directamente (SEGMENTS, XSENS_NAME_MAP)
    listos para usar. Lanza un error claro (no un fallo críptico a mitad
    de camino) si ningún protocolo conocido calza con los marcadores del
    archivo, listando qué le faltó a cada candidato para que sea fácil
    diagnosticar si es un protocolo nuevo o un error de nombres.
    """
    labels_set = set(labels_c3d)
    detectado = buscar_protocolo(labels_set)

    if detectado is None:
        mensaje = (
            "No se pudo detectar el protocolo de marcadores de este c3d.\n"
            "Marcadores en el archivo:\n  " + ", ".join(sorted(labels_set)) + "\n\n"
            "Lo que le faltó a cada protocolo conocido:\n"
        )
        for nombre_modulo in PROTOCOLOS_CONOCIDOS:
            modulo = importlib.import_module(nombre_modulo)
            requeridos = set(modulo.MARCADORES_REQUERIDOS)
            faltantes = sorted(requeridos - labels_set)
            mensaje += f"  {nombre_modulo}: faltan {faltantes}\n"
        mensaje += (
            "\nSi este es un protocolo nuevo, crear segments_protocolo_NUEVO.py "
            "(ver detectar_protocolo.py) y agregarlo a PROTOCOLOS_CONOCIDOS."
        )
        raise ValueError(mensaje)

    if verbose:
        extra = " (coincidencia exacta)" if detectado.coincidencia_exacta else " (subconjunto compatible, el c3d tiene marcadores de más)"
        print(f"Protocolo detectado: {detectado.nombre_modulo}{extra}")

    return cargar_segments(detectado.nombre_modulo)
