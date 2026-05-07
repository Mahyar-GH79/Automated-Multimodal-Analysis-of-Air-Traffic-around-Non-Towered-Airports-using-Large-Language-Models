#!/usr/bin/env python3
"""visualize_trajectories.py — 3D ADS-B Trajectory Viewer for CTAF-KHAF dataset"""

import json, math, sys, os, webbrowser, tempfile
from pathlib import Path

KHAF_LAT, KHAF_LON = 37.5134, -122.5008

def to_xyz(lat, lon, alt_ft):
    dx = (lon - KHAF_LON) * math.cos(math.radians(KHAF_LAT)) * 60
    dy = (lat - KHAF_LAT) * 60
    dz = alt_ft / 6076.0
    return round(dx, 4), round(dy, 4), round(dz, 4)

def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    scenarios = []
    for sc in data['scenarios']:
        tracks = []
        for cs, track in sc['adsb_trajectories'].items():
            xs, ys, zs, ts, phases = [], [], [], [], []
            for p in track:
                x, y, z = to_xyz(p['lat'], p['lon'], p['alt_ft'])
                xs.append(x); ys.append(y); zs.append(z)
                ts.append(p['time_s'])
                phases.append(p.get('phase', ''))
            tracks.append({
                'callsign': cs,
                'type': track[0]['aircraft_type'],
                'has_radio': track[0]['has_radio'],
                'on_ground': [p.get('on_ground', False) for p in track],
                'x': xs, 'y': ys, 'z': zs,
                't': ts, 'phase': phases,
            })
        tx = sc.get('transcript_ground_truth', [])
        scenarios.append({
            'id': sc['scenario_id'],
            'label': sc['label'],
            'hazard_type': sc['hazard_type'],
            'tracks': tracks,
            'metar': sc['metar']['raw'],
            'advisory': sc.get('ground_truth_advisory', '')[:300],
            'tx': [e['text'][:90] for e in tx],
            'tx_ts': [e['timestamp'].split('-->')[0].strip() for e in tx],
        })
    return scenarios

def build_html(scenarios):
    data_json = json.dumps(scenarios)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CTAF-KHAF 3D Trajectory Viewer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f1117; color: #e2e8f0; height: 100vh; display: flex; flex-direction: column; }}
#header {{ padding: 10px 16px; background: #1a1d2e; border-bottom: 1px solid #2d3148;
           display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
#header h1 {{ font-size: 15px; font-weight: 600; color: #a5b4fc; letter-spacing: 0.5px; }}
#controls {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
select, input {{ background: #252836; border: 1px solid #3d4166; color: #e2e8f0;
                 padding: 5px 10px; border-radius: 6px; font-size: 13px; cursor: pointer; }}
select:hover, input:hover {{ border-color: #6366f1; }}
.badge {{ padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;
          text-transform: uppercase; letter-spacing: 0.5px; }}
.hazard  {{ background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }}
.warning {{ background: #431407; color: #fb923c; border: 1px solid #7c2d12; }}
.nominal {{ background: #052e16; color: #4ade80; border: 1px solid #14532d; }}
#main {{ display: flex; flex: 1; overflow: hidden; }}
#plot {{ flex: 1; }}
#sidebar {{ width: 300px; background: #1a1d2e; border-left: 1px solid #2d3148;
            overflow-y: auto; padding: 12px; flex-shrink: 0; }}
#sidebar h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
               color: #6366f1; margin-bottom: 8px; }}
.info-block {{ background: #252836; border-radius: 8px; padding: 10px; margin-bottom: 10px;
               border: 1px solid #2d3148; }}
.info-label {{ font-size: 10px; color: #64748b; text-transform: uppercase;
               letter-spacing: 0.5px; margin-bottom: 3px; }}
.info-val {{ font-size: 12px; color: #cbd5e1; line-height: 1.5; }}
.tx-line {{ font-size: 11px; color: #94a3b8; padding: 4px 0;
             border-bottom: 1px solid #1e2235; line-height: 1.4; }}
.tx-ts {{ color: #6366f1; font-size: 10px; }}
.ac-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }}
#sclist {{ max-height: 200px; overflow-y: auto; margin-top: 6px; }}
.sc-item {{ font-size: 12px; padding: 5px 8px; border-radius: 5px; cursor: pointer;
             display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
.sc-item:hover {{ background: #252836; }}
.sc-item.active {{ background: #2d3148; border-left: 3px solid #6366f1; padding-left: 5px; }}
.sc-id {{ color: #6366f1; font-weight: 600; min-width: 36px; }}
.sc-type {{ color: #94a3b8; font-size: 10px; flex: 1; }}
#nav {{ display: flex; gap: 6px; }}
button {{ background: #252836; border: 1px solid #3d4166; color: #e2e8f0;
          padding: 5px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; }}
button:hover {{ background: #2d3148; border-color: #6366f1; color: #a5b4fc; }}
#search {{ width: 160px; }}
</style>
</head>
<body>
<div id="header">
  <h1>CTAF-KHAF · 3D Trajectory Viewer</h1>
  <div id="controls">
    <div id="nav">
      <button onclick="navigate(-1)">&#8592; Prev</button>
      <button onclick="navigate(1)">Next &#8594;</button>
    </div>
    <input id="search" type="text" placeholder="Search scenario..." oninput="filterList()">
    <select id="labelFilter" onchange="filterList()">
      <option value="">All labels</option>
      <option value="hazard">Hazard</option>
      <option value="warning">Warning</option>
      <option value="nominal">Nominal</option>
    </select>
    <select id="typeFilter" onchange="filterList()">
      <option value="">All types</option>
    </select>
  </div>
  <span id="cur-badge" class="badge"></span>
</div>
<div id="main">
  <div id="plot"></div>
  <div id="sidebar">
    <h2>Scenario</h2>
    <div class="info-block">
      <div class="info-label">ID / Type</div>
      <div class="info-val" id="s-id">—</div>
      <div class="info-label" style="margin-top:6px;">METAR</div>
      <div class="info-val" id="s-metar" style="font-size:11px;">—</div>
    </div>
    <div class="info-block">
      <div class="info-label">Advisory</div>
      <div class="info-val" id="s-adv" style="font-size:11px;color:#94a3b8;">—</div>
    </div>
    <div class="info-block">
      <div class="info-label">Aircraft</div>
      <div id="s-ac" class="info-val">—</div>
    </div>
    <div class="info-block">
      <div class="info-label">Transcript</div>
      <div id="s-tx"></div>
    </div>
    <h2 style="margin-top:12px;">All Scenarios</h2>
    <div id="sclist"></div>
  </div>
</div>

<script>
const SCENARIOS = {data_json};
const COLORS = ['#818cf8','#f87171','#34d399','#fbbf24','#60a5fa','#e879f9'];
let currentIdx = 0;
let filtered = SCENARIOS.map((_,i)=>i);

const AC_COLORS = {{}};
SCENARIOS.forEach(sc => {{
  sc.tracks.forEach((t, i) => {{
    if (!AC_COLORS[t.callsign]) AC_COLORS[t.callsign] = COLORS[i % COLORS.length];
  }});
}});

function plot(idx) {{
  currentIdx = idx;
  const sc = SCENARIOS[idx];
  const traces = [];

  // Runway 30 centerline (NW direction from KHAF)
  const rwLen = 3.5;
  const hdg = 300 * Math.PI / 180;
  const rwx = [-Math.sin(hdg)*rwLen, Math.sin(hdg)*rwLen];
  const rwy = [-Math.cos(hdg)*rwLen, Math.cos(hdg)*rwLen];
  traces.push({{
    type: 'scatter3d', mode: 'lines',
    x: rwx, y: rwy, z: [0, 0],
    line: {{ color: '#4ade80', width: 3, dash: 'dash' }},
    name: 'Rwy 30 centerline',
    hoverinfo: 'none',
  }});

  // KHAF airport marker
  traces.push({{
    type: 'scatter3d', mode: 'markers+text',
    x: [0], y: [0], z: [0.011],
    marker: {{ size: 8, color: '#4ade80', symbol: 'diamond' }},
    text: ['KHAF'], textfont: {{ color: '#4ade80', size: 11 }},
    textposition: 'top center',
    name: 'KHAF', hoverinfo: 'name',
  }});

  sc.tracks.forEach((t, ti) => {{
    const col = COLORS[ti % COLORS.length];
    const label = t.callsign + (t.has_radio ? '' : ' (NORDO)');

    // Main 3D track
    traces.push({{
      type: 'scatter3d', mode: 'lines+markers',
      x: t.x, y: t.y, z: t.z,
      line: {{ color: col, width: 3 }},
      marker: {{ size: 3, color: col }},
      name: label,
      text: t.t.map((ts, i) => `${{t.callsign}}<br>t=${{ts}}s<br>alt=${{Math.round(t.z[i]*6076)}}ft<br>${{t.phase[i]}}`),
      hovertemplate: '%{{text}}<extra></extra>',
    }});

    // Start marker
    traces.push({{
      type: 'scatter3d', mode: 'markers+text',
      x: [t.x[0]], y: [t.y[0]], z: [t.z[0]],
      marker: {{ size: 7, color: col, symbol: 'circle', line: {{ color: '#fff', width: 1 }} }},
      text: [t.callsign], textfont: {{ color: col, size: 10 }},
      textposition: ti === 0 ? 'top right' : 'bottom left',
      name: '', showlegend: false, hoverinfo: 'skip',
    }});

    // Altitude shadow (projection onto ground plane)
    traces.push({{
      type: 'scatter3d', mode: 'lines',
      x: t.x, y: t.y, z: t.z.map(() => 0),
      line: {{ color: col, width: 1, dash: 'dot' }},
      opacity: 0.25, name: '', showlegend: false, hoverinfo: 'skip',
    }});

    // Vertical drop lines at key points (every 3rd point)
    const dropX=[], dropY=[], dropZ=[];
    t.x.forEach((x, i) => {{
      if (i % 4 === 0) {{
        dropX.push(x, x, null);
        dropY.push(t.y[i], t.y[i], null);
        dropZ.push(t.z[i], 0, null);
      }}
    }});
    traces.push({{
      type: 'scatter3d', mode: 'lines',
      x: dropX, y: dropY, z: dropZ,
      line: {{ color: col, width: 0.5 }},
      opacity: 0.2, name: '', showlegend: false, hoverinfo: 'skip',
    }});
  }});

  const layout = {{
    paper_bgcolor: '#0f1117',
    plot_bgcolor:  '#0f1117',
    margin: {{ l: 0, r: 0, t: 30, b: 0 }},
    title: {{
      text: `${{sc.id}} — ${{sc.hazard_type.replace(/_/g,' ')}}`,
      font: {{ color: '#a5b4fc', size: 14 }},
      x: 0.02,
    }},
    scene: {{
      xaxis: {{ title: 'East (NM)', color: '#475569', gridcolor: '#1e2235', zerolinecolor: '#334155', titlefont: {{ size: 11 }} }},
      yaxis: {{ title: 'North (NM)', color: '#475569', gridcolor: '#1e2235', zerolinecolor: '#334155', titlefont: {{ size: 11 }} }},
      zaxis: {{ title: 'Alt (NM)', color: '#475569', gridcolor: '#1e2235', zerolinecolor: '#334155', titlefont: {{ size: 11 }} }},
      bgcolor: '#0f1117',
      camera: {{ eye: {{ x: -1.8, y: -1.8, z: 1.2 }} }},
      aspectmode: 'manual',
      aspectratio: {{ x: 1.5, y: 1.5, z: 0.4 }},
    }},
    legend: {{ font: {{ color: '#94a3b8', size: 11 }}, bgcolor: 'rgba(0,0,0,0)', x: 0, y: 1 }},
    showlegend: true,
  }};

  Plotly.react('plot', traces, layout, {{responsive: true, displayModeBar: false}});
  updateSidebar(sc);
  highlightList(idx);
}};

function updateSidebar(sc) {{
  document.getElementById('s-id').innerHTML = `<strong style="color:#a5b4fc">${{sc.id}}</strong> &nbsp;${{sc.hazard_type.replace(/_/g,' ')}}`;
  document.getElementById('s-metar').textContent = sc.metar;
  document.getElementById('s-adv').textContent = sc.advisory;
  document.getElementById('cur-badge').textContent = sc.label;
  document.getElementById('cur-badge').className = 'badge ' + sc.label;

  const acDiv = document.getElementById('s-ac');
  acDiv.innerHTML = sc.tracks.map((t,i) =>
    `<div style="margin-bottom:3px;">
      <span class="ac-dot" style="background:${{COLORS[i%COLORS.length]}}"></span>
      <strong>${{t.callsign}}</strong> · ${{t.type}}${{t.has_radio ? '' : ' <em style="color:#f87171">(NORDO)</em>'}}
     </div>`
  ).join('');

  const txDiv = document.getElementById('s-tx');
  txDiv.innerHTML = sc.tx.map((line, i) =>
    `<div class="tx-line"><span class="tx-ts">${{sc.tx_ts[i]}}</span><br>${{line}}</div>`
  ).join('');
}}

function buildList() {{
  const types = [...new Set(SCENARIOS.map(s => s.hazard_type))].sort();
  const sel = document.getElementById('typeFilter');
  types.forEach(t => {{
    const o = document.createElement('option');
    o.value = t; o.textContent = t.replace(/_/g, ' ');
    sel.appendChild(o);
  }});
  renderList();
}}

function filterList() {{
  const q = document.getElementById('search').value.toLowerCase();
  const lb = document.getElementById('labelFilter').value;
  const tp = document.getElementById('typeFilter').value;
  filtered = SCENARIOS.map((_,i)=>i).filter(i => {{
    const s = SCENARIOS[i];
    if (lb && s.label !== lb) return false;
    if (tp && s.hazard_type !== tp) return false;
    if (q && !s.id.toLowerCase().includes(q) && !s.hazard_type.includes(q)) return false;
    return true;
  }});
  renderList();
}}

function renderList() {{
  const div = document.getElementById('sclist');
  div.innerHTML = filtered.map(i => {{
    const s = SCENARIOS[i];
    return `<div class="sc-item${{i===currentIdx?' active':''}}" onclick="plot(${{i}})" id="li${{i}}">
      <span class="sc-id">${{s.id}}</span>
      <span class="sc-type">${{s.hazard_type.replace(/_/g,' ')}}</span>
      <span class="badge ${{s.label}}">${{s.label[0].toUpperCase()}}</span>
    </div>`;
  }}).join('');
}}

function highlightList(idx) {{
  document.querySelectorAll('.sc-item').forEach(el => el.classList.remove('active'));
  const el = document.getElementById('li'+idx);
  if (el) {{ el.classList.add('active'); el.scrollIntoView({{block:'nearest'}}); }}
}}

function navigate(dir) {{
  const pos = filtered.indexOf(currentIdx);
  const next = filtered[(pos + dir + filtered.length) % filtered.length];
  plot(next);
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft')  navigate(-1);
}});

buildList();
plot(0);
</script>
</body>
</html>"""

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'dataset_v2/ctaf_khaf_v2.json'
    print(f"Loading {path}...")
    scenarios = load_dataset(path)
    print(f"Loaded {len(scenarios)} scenarios")
    html = build_html(scenarios)
    tmp = tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w')
    tmp.write(html)
    tmp.close()
    print(f"Opening viewer: {tmp.name}")
    webbrowser.open(f'file://{tmp.name}')
    print("Done. Press Ctrl+C to exit (file stays open in browser).")