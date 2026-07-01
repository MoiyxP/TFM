"""
generar_visualizador.py
=========================
Genera el visor 3D interactivo (marcadores + esqueleto + ejes locales de
los 7 segmentos) a partir de un c3d y una definición de segmentos, en un
solo comando. Pensado para correr cada vez que cambies algo en
segments_protocolo_94.py (o el protocolo que uses) y quieras confirmar
visualmente el resultado, sin depender de nadie más.

USO:
    1. Ajustá las 3 variables de la sección "CONFIGURACIÓN" más abajo.
    2. Corré: python generar_visualizador.py
    3. Abrí el .html generado en cualquier navegador.

QUÉ TOCAR SI ALGO SE VE MAL:
    - Si un eje/segmento apunta en la dirección física incorrecta
      (ej. verde hacia abajo en vez de hacia arriba): el problema está en
      segments_protocolo_94.py (la definición de vectores), NO en este
      script. Corregí ahí y volvé a correr este script.
    - Si las flechas se ven muy largas/cortas o se superponen entre
      segmentos cercanos (ej. tobillo y talón): cambiá LARGO_FLECHA_MM
      más abajo, no hace falta tocar nada más.
    - Si querés agregar/quitar líneas del esqueleto: editá la lista
      HUESOS más abajo (pares de nombres de marcador que se conectan).
"""

import sys
import json
import numpy as np

# =============================================================================
# CONFIGURACIÓN - ajustar estas 3 líneas según lo que quieras visualizar
# =============================================================================

RUTA_MAIN = "/home/claude/work2"                                    # carpeta donde está main.py
C3D_A_VISUALIZAR = "/mnt/user-data/uploads/Sub01_Walk001.c3d"        # qué trial mostrar
ARCHIVO_SALIDA = "visualizacion_3d.html"                             # nombre del html generado

LARGO_FLECHA_MM = 45     # largo de las flechas de ejes locales, en mm
PASO_SUBMUESTREO = 2     # 1=todos los frames, 2=uno de cada dos (más liviano/rápido), etc.

# =============================================================================
# Esqueleto: pares de marcadores que se conectan con una línea. Ajustar
# aquí si tu protocolo tiene otros nombres de marcador o querés más/menos
# líneas. No afecta ningún cálculo, es solo dibujo.
# =============================================================================

HUESOS = [
    ["LFHD", "RFHD"], ["LBHD", "RBHD"], ["LFHD", "LBHD"], ["RFHD", "RBHD"],
    ["CTRHD", "LFHD"], ["CTRHD", "RFHD"],
    ["CTRHD", "CV7"], ["CV7", "SJN"], ["CV7", "TV4"], ["SJN", "SXS"],
    ["CV7", "LSHO"], ["CV7", "RSHO"], ["LSHO", "LCAJ"], ["RSHO", "RCAJ"],
    ["SJN", "LSHO"], ["SJN", "RSHO"],
    ["LIAS", "RIAS"], ["LIAS", "LIPS"], ["RIAS", "RIPS"], ["LIPS", "RIPS"],
    ["LIPS", "SCRM"], ["RIPS", "SCRM"], ["LIAS", "SXS"], ["RIAS", "SXS"],
    ["LSHO", "LHLE"], ["LSHO", "LHME"], ["LHLE", "LHME"],
    ["LHLE", "LULB"], ["LHME", "LULB"], ["LULB", "LLLB"],
    ["LLLB", "LRSP"], ["LLLB", "LUSP"], ["LRSP", "LUSP"],
    ["LRSP", "LHND"], ["LUSP", "LHND"], ["LHND", "LHL2"], ["LHND", "LHM5"],
    ["RSHO", "RHLE"], ["RSHO", "RHME"], ["RHLE", "RHME"],
    ["RHLE", "RULB"], ["RHME", "RULB"], ["RULB", "RLLB"],
    ["RLLB", "RRSP"], ["RLLB", "RUSP"], ["RRSP", "RUSP"],
    ["RRSP", "RHND"], ["RUSP", "RHND"], ["RHND", "RHL2"], ["RHND", "RHM5"],
    ["LIAS", "LFLE"], ["LIPS", "LFLE"], ["LIAS", "LFME"], ["LICT", "LFLE"],
    ["LFLE", "LFME"], ["LFLE", "LTTC"], ["LFME", "LTTC"],
    ["LTTC", "LFAL"], ["LFLE", "LFAL"], ["LFAL", "LTAM"],
    ["LFAL", "LFCC"], ["LTAM", "LFCC"], ["LFCC", "LTOE"],
    ["LFCC", "L1MH"], ["LFCC", "L5MH"], ["L1MH", "L5MH"], ["L1MH", "LTOE"],
    ["RIAS", "RFLE"], ["RIPS", "RFLE"], ["RIAS", "RFME"], ["RICT", "RFLE"],
    ["RFLE", "RFME"], ["RFLE", "RTTC"], ["RFME", "RTTC"],
    ["RTTC", "RFAL"], ["RFLE", "RFAL"], ["RFAL", "RTAM"],
    ["RFAL", "RFCC"], ["RTAM", "RFCC"], ["RFCC", "RTOE"],
    ["RFCC", "R1MH"], ["RFCC", "R5MH"], ["R1MH", "R5MH"], ["R1MH", "RTOE"],
]


# =============================================================================
# A partir de acá no debería hacer falta tocar nada - es el motor que arma
# los datos y el HTML. Si tu protocolo tiene otro nombre de módulo de
# segmentos, cambiá el import de más abajo.
# =============================================================================

def generar():
    sys.path.insert(0, RUTA_MAIN)
    import Project as main
    import segments_protocolo_94 as seg94  # <- cambiar si el módulo se llama distinto

    main.SEGMENTS = seg94.SEGMENTS

    print(f"Leyendo {C3D_A_VISUALIZAR} ...")
    markers, analogs = main.read_c3d(C3D_A_VISUALIZAR)
    int_markers, _ = main.interpolate_c3d(markers, analogs, "akima")

    sub = int_markers.isel(time=slice(0, None, PASO_SUBMUESTREO))
    labels = list(sub.channel.values)

    # --- 1. Posiciones de marcadores por frame ---
    xyz = sub.values[:3, :, :]
    n_frames = xyz.shape[2]
    frames_data = []
    for f in range(n_frames):
        frame_pts = []
        for m in range(len(labels)):
            x, y, z = xyz[:, m, f]
            if np.isnan(x):
                frame_pts.append(None)
            else:
                frame_pts.append([round(float(x), 1), round(float(y), 1), round(float(z), 1)])
        frames_data.append(frame_pts)
    markers3d = {"labels": labels, "frames": frames_data}
    print(f"  {n_frames} frames, {len(labels)} marcadores")

    # --- 2. Ejes locales de cada segmento por frame ---
    def origen_promedio(markers, nombres):
        pts = [markers.sel(channel=[n]).values[:3, 0, :] for n in nombres]
        return np.mean(pts, axis=0)

    axes_data = {}
    for seg_name, seg_def in seg94.SEGMENTS.items():
        R = main.construir_matriz_rotacion(sub, seg_def)
        origin = origen_promedio(sub, seg_def.origin)
        frames_axes = []
        for f in range(R.shape[2]):
            o = origin[:, f]
            if np.isnan(o).any() or np.isnan(R[:, :, f]).any():
                frames_axes.append(None)
                continue
            x_end = o + R[:, 0, f] * LARGO_FLECHA_MM
            y_end = o + R[:, 1, f] * LARGO_FLECHA_MM
            z_end = o + R[:, 2, f] * LARGO_FLECHA_MM
            frames_axes.append({
                "origin": [round(float(v), 1) for v in o],
                "x_end": [round(float(v), 1) for v in x_end],
                "y_end": [round(float(v), 1) for v in y_end],
                "z_end": [round(float(v), 1) for v in z_end],
            })
        axes_data[seg_name] = frames_axes
    print(f"  Ejes calculados para: {', '.join(axes_data.keys())}")

    # --- 3. Verificar que los huesos definidos existen en el archivo ---
    faltantes = {m for par in HUESOS for m in par if m not in labels}
    if faltantes:
        print(f"  AVISO: estos marcadores de HUESOS no existen en el c3d y se van a omitir: {faltantes}")

    # --- 4. Armar el HTML ---
    html = _plantilla_html()
    html = html.replace("__NFRAMES__", str(n_frames))
    html = html.replace("__MAXFRAME__", str(n_frames - 1))
    html = html.replace("__MARKERS_JSON__", json.dumps(markers3d))
    html = html.replace("__BONES_JSON__", json.dumps(HUESOS))
    html = html.replace("__AXES_JSON__", json.dumps(axes_data))

    with open(ARCHIVO_SALIDA, "w") as f:
        f.write(html)

    import os
    print(f"\nListo: {ARCHIVO_SALIDA} ({round(os.path.getsize(ARCHIVO_SALIDA)/1024, 1)} KB)")
    print("Abrilo con cualquier navegador.")


def _plantilla_html():
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Visualizacion 3D</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<style>
  body { font-family: -apple-system, sans-serif; margin: 0; padding: 16px; background: #fafafa; }
  #controls { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }
  #plot { width: 100%; height: 720px; }
  button { padding: 6px 14px; cursor: pointer; }
  input[type=range] { flex: 1; min-width: 200px; }
  label { font-size: 14px; color: #333; }
  .legend-item { display: inline-flex; align-items: center; gap: 4px; font-size: 13px; margin-right: 14px; }
  .swatch { width: 14px; height: 4px; display: inline-block; }
</style>
</head>
<body>
<h2>Visualizacion 3D - marcadores, esqueleto y ejes locales</h2>
<div id="controls">
  <button id="playBtn">Reproducir</button>
  <label>Frame: <span id="frameLabel">0</span> / __NFRAMES__</label>
  <input type="range" id="frameSlider" min="0" max="__MAXFRAME__" value="0" step="1">
  <label>Velocidad:
    <select id="speedSel">
      <option value="200">0.5x</option>
      <option value="100" selected>1x</option>
      <option value="50">2x</option>
      <option value="25">4x</option>
    </select>
  </label>
</div>
<div style="margin-bottom:8px;">
  <span class="legend-item"><span class="swatch" style="background:#d62728;"></span>Eje X local (anterior)</span>
  <span class="legend-item"><span class="swatch" style="background:#2ca02c;"></span>Eje Y local (vertical / longitudinal)</span>
  <span class="legend-item"><span class="swatch" style="background:#1f77b4;"></span>Eje Z local (lateral)</span>
</div>
<div id="plot"></div>

<script>
const markersData = __MARKERS_JSON__;
const bonesList = __BONES_JSON__;
const axesData = __AXES_JSON__;

const labels = markersData.labels;
const frames = markersData.frames;
const nFrames = frames.length;

document.getElementById("frameSlider").max = nFrames - 1;

function labelIndex(name) { return labels.indexOf(name); }

function buildTraces(frameIdx) {
  const pts = frames[frameIdx];

  const mx = [], my = [], mz = [], mtext = [];
  pts.forEach((p, i) => {
    if (p) { mx.push(p[0]); my.push(p[1]); mz.push(p[2]); mtext.push(labels[i]); }
  });
  const markerTrace = {
    type: "scatter3d", mode: "markers",
    x: mx, y: my, z: mz, text: mtext,
    marker: { size: 3, color: "#444" },
    name: "Marcadores", hoverinfo: "text"
  };

  const boneX = [], boneY = [], boneZ = [];
  bonesList.forEach(([a, b]) => {
    const ia = labelIndex(a), ib = labelIndex(b);
    const pa = pts[ia], pb = pts[ib];
    if (pa && pb) {
      boneX.push(pa[0], pb[0], null);
      boneY.push(pa[1], pb[1], null);
      boneZ.push(pa[2], pb[2], null);
    }
  });
  const boneTrace = {
    type: "scatter3d", mode: "lines",
    x: boneX, y: boneY, z: boneZ,
    line: { color: "#999", width: 2 },
    name: "Esqueleto", hoverinfo: "skip"
  };

  const axisTraces = [];
  const colors = { x: "#d62728", y: "#2ca02c", z: "#1f77b4" };
  for (const segName in axesData) {
    const seg = axesData[segName][frameIdx];
    if (!seg) continue;
    for (const axKey of ["x", "y", "z"]) {
      const end = seg[axKey + "_end"];
      axisTraces.push({
        type: "scatter3d", mode: "lines",
        x: [seg.origin[0], end[0]],
        y: [seg.origin[1], end[1]],
        z: [seg.origin[2], end[2]],
        line: { color: colors[axKey], width: 5 },
        showlegend: false, hoverinfo: "skip"
      });
    }
  }

  return [markerTrace, boneTrace, ...axisTraces];
}

const layout = {
  scene: {
    xaxis: { title: "X lab (mm)" },
    yaxis: { title: "Y lab (mm)" },
    zaxis: { title: "Z lab (mm)" },
    aspectmode: "data",
    camera: { eye: { x: 1.6, y: 1.6, z: 0.8 } }
  },
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: false
};

Plotly.newPlot("plot", buildTraces(0), layout, { responsive: true });

let playing = false, timer = null, currentFrame = 0;

function setFrame(f) {
  currentFrame = f;
  document.getElementById("frameSlider").value = f;
  document.getElementById("frameLabel").textContent = f;
  Plotly.react("plot", buildTraces(f), layout, { responsive: true });
}

document.getElementById("frameSlider").addEventListener("input", (e) => setFrame(parseInt(e.target.value)));

document.getElementById("playBtn").addEventListener("click", () => {
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "Pausar" : "Reproducir";
  if (playing) {
    const speed = parseInt(document.getElementById("speedSel").value);
    timer = setInterval(() => { currentFrame = (currentFrame + 1) % nFrames; setFrame(currentFrame); }, speed);
  } else {
    clearInterval(timer);
  }
});

document.getElementById("speedSel").addEventListener("change", () => {
  if (playing) {
    clearInterval(timer);
    const speed = parseInt(document.getElementById("speedSel").value);
    timer = setInterval(() => { currentFrame = (currentFrame + 1) % nFrames; setFrame(currentFrame); }, speed);
  }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    generar()
