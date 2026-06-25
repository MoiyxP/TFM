import Project as m
import detectar_protocolo

markers, _ = m.read_c3d(m.movement_c3d)
m.SEGMENTS, m.XSENS_NAME_MAP = detectar_protocolo.detectar_y_cargar(markers.channel.values)
int_markers, _ = m.interpolate_c3d(markers, None, 'akima')

import localizar_marcador_culpable as loc
loc.inspeccionar_marcadores_pv(int_markers, frames_sospechosos=[282, 353], ventana=3)