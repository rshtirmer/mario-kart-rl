"""Standalone dashboard server -- monitors training runs and serves web UI."""

import json
import os
from pathlib import Path

from aiohttp import web

# Project root is 3 levels up from this file: src/mariokart/dashboard.py -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = _PROJECT_ROOT / "runs"
RESULTS_FILE = _PROJECT_ROOT / "results.tsv"
RECORDS_FILE = _PROJECT_ROOT / "records.json"

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MARIO KART RL</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#000000;--bg2:#080808;--bg3:#111111;
  --border:#1a1a1a;
  --white:#ffffff;
  --green:#00ff88;--red:#ff3355;--blue:#4488ff;--cyan:#00ddff;--purple:#aa77ff;
  --glow-g:0 0 20px rgba(0,255,136,0.3);--glow-b:0 0 20px rgba(68,136,255,0.3);
  --glow-c:0 0 20px rgba(0,221,255,0.3);
  --mono:'SF Mono','JetBrains Mono','Fira Code',Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,'Inter',system-ui,sans-serif;
}
html{font-size:14px}
body{background:#000;color:#fff;font-family:var(--sans);overflow-x:hidden;min-height:100vh}

.header{padding:20px 28px;display:flex;align-items:center;gap:16px;border-bottom:1px solid var(--border)}
.header h1{font-family:var(--mono);font-size:16px;font-weight:700;letter-spacing:4px;color:#fff}
.header .dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:var(--glow-g);animation:pulse 2s infinite}
.header .status{font-size:12px;color:#fff;font-family:var(--mono);opacity:0.6}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)}
.stat{background:var(--bg2);padding:10px 16px}
.stat .label{font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#fff;opacity:0.45;margin-bottom:4px;font-family:var(--mono)}
.stat .value{font-size:20px;font-weight:700;font-family:var(--mono);color:#fff}
.stat .value.green{color:var(--green);text-shadow:var(--glow-g)}
.stat .value.cyan{color:var(--cyan);text-shadow:var(--glow-c)}
.stat .value.blue{color:var(--blue);text-shadow:var(--glow-b)}

.main{display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-rows:1fr 1fr auto;gap:1px;background:var(--border);height:calc(100vh - 85px)}
.panel{background:var(--bg2);padding:12px 16px;overflow:hidden}
.panel-title{font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#fff;opacity:0.5;font-family:var(--mono);margin-bottom:8px;display:flex;align-items:center;gap:6px}
.panel-title::before{content:'';width:3px;height:10px;background:var(--cyan);border-radius:1px;box-shadow:var(--glow-c)}

canvas{width:100%;height:100px;display:block;margin-bottom:4px}

.live-grid{display:grid;grid-template-columns:repeat(4,1fr);grid-template-rows:1fr 1fr;gap:2px;height:100%}
.live-cell{position:relative;background:#000;overflow:hidden}
.live-cell img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}
.live-cell .env-id{position:absolute;top:2px;left:4px;font-size:8px;font-family:var(--mono);color:var(--cyan);opacity:0.7}
.live-badge{position:absolute;top:8px;right:8px;background:var(--red);color:#fff;font-size:8px;font-family:var(--mono);padding:1px 6px;border-radius:2px;animation:pulse 1.5s infinite;letter-spacing:1px;z-index:1}

/* Fullscreen overlay */
.fullscreen-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.95);z-index:1000;align-items:center;justify-content:center;flex-direction:column}
.fullscreen-overlay.show{display:flex}
.fullscreen-overlay img{max-width:90vw;max-height:80vh;image-rendering:pixelated;border:1px solid var(--border)}
.fullscreen-overlay .controls{display:flex;align-items:center;gap:16px;margin-top:16px}
.fullscreen-overlay .controls button{background:none;border:1px solid rgba(255,255,255,0.3);color:#fff;padding:8px 20px;border-radius:4px;cursor:pointer;font-family:var(--mono);font-size:13px;transition:border-color 0.15s}
.fullscreen-overlay .controls button:hover{border-color:var(--cyan);color:var(--cyan)}
.fullscreen-overlay .step-text{color:#fff;font-family:var(--mono);font-size:13px;opacity:0.7}
.fullscreen-overlay .close-btn{position:absolute;top:20px;right:24px;background:none;border:none;color:#fff;font-size:28px;cursor:pointer;opacity:0.5}
.fullscreen-overlay .close-btn:hover{opacity:1}

.wr-item{margin-bottom:12px}
.wr-name{font-size:12px;font-family:var(--mono);color:#fff;margin-bottom:4px;display:flex;justify-content:space-between}
.wr-bar{height:5px;background:#111;border-radius:3px;overflow:hidden}
.wr-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));border-radius:3px;box-shadow:var(--glow-c);transition:width 0.5s}

.experiments{grid-column:1/-1;background:var(--bg2);padding:20px 24px;border-top:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:13px}
thead th{text-align:left;padding:10px 14px;color:#fff;opacity:0.45;font-size:11px;text-transform:uppercase;letter-spacing:1.5px;border-bottom:1px solid var(--border);font-weight:500}
tbody td{padding:8px 14px;color:#fff;border-bottom:1px solid rgba(26,26,26,0.6)}
tbody tr:hover{background:var(--bg3)}
.tag{display:inline-block;padding:3px 10px;border-radius:3px;font-size:11px;font-weight:600}
.tag.keep{background:rgba(0,255,136,0.15);color:var(--green);border:1px solid rgba(0,255,136,0.25)}
.tag.crash{background:rgba(255,51,85,0.12);color:var(--red);border:1px solid rgba(255,51,85,0.2)}
.tag.discard{background:rgba(122,131,155,0.12);color:#fff;opacity:0.5;border:1px solid rgba(122,131,155,0.2)}

@media(max-width:900px){.stats{grid-template-columns:repeat(3,1fr)}.main{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="header">
  <h1>MARIO KART RL</h1>
  <div class="dot"></div>
  <span class="status" id="status">connecting...</span>
</div>

<div class="stats">
  <div class="stat"><div class="label">Steps</div><div class="value cyan" id="s-steps">--</div></div>
  <div class="stat"><div class="label">Episodes</div><div class="value" id="s-episodes">--</div></div>
  <div class="stat"><div class="label">Best Lap</div><div class="value green" id="s-bestlap">--</div></div>
  <div class="stat"><div class="label">FPS</div><div class="value blue" id="s-fps">--</div></div>
  <div class="stat"><div class="label">Memory</div><div class="value" id="s-mem">--</div></div>
  <div class="stat"><div class="label">Elapsed</div><div class="value" id="s-elapsed">--</div></div>
</div>

<div class="main">
  <div class="panel">
    <div class="panel-title">Reward</div>
    <canvas id="c-reward"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">Lap Times</div>
    <canvas id="c-lap"></canvas>
  </div>
  <div class="panel" style="grid-row:1/3;position:relative">
    <div class="panel-title">Live Agents (8 envs)</div>
    <div class="live-badge">LIVE</div>
    <div class="live-grid" id="live-grid"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Loss</div>
    <canvas id="c-loss"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">WR Progress</div>
    <div id="wr-tracker"></div>
  </div>
  <div class="experiments">
    <div class="panel-title">Experiments</div>
    <table>
      <thead><tr><th>Commit</th><th>Lap</th><th>WR</th><th>Status</th><th>Description</th></tr></thead>
      <tbody id="exp-body"></tbody>
    </table>
  </div>
</div>

<div class="fullscreen-overlay" id="fs-overlay">
  <button class="close-btn" onclick="closeFullscreen()">&times;</button>
  <img id="fs-img" src="">
  <div class="controls">
    <button onclick="fsNav(-1)">&larr; Prev</button>
    <span class="step-text" id="fs-step">--</span>
    <button onclick="fsNav(1)">Next &rarr;</button>
  </div>
</div>

<script>
const DPR = window.devicePixelRatio || 1;

function initCanvas(id) {
  const c = document.getElementById(id);
  if (!c) return null;
  const r = c.getBoundingClientRect();
  c.width = r.width * DPR;
  c.height = r.height * DPR;
  return c;
}

function drawChart(id, data, color, opts = {}) {
  const c = initCanvas(id);
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  if (!data.length) return;

  const pad = {l: 50 * DPR, r: 10 * DPR, t: 8 * DPR, b: 20 * DPR};
  const w = W - pad.l - pad.r, h = H - pad.t - pad.b;

  // Smooth data (moving average)
  const smooth = opts.smooth || 5;
  const sd = data.map((_, i) => {
    const start = Math.max(0, i - smooth + 1);
    const slice = data.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });

  let minY = Math.min(...sd), maxY = Math.max(...sd);
  if (opts.minY !== undefined) minY = Math.min(minY, opts.minY);
  const rangeY = maxY - minY || 1;
  const toX = i => pad.l + (i / (sd.length - 1 || 1)) * w;
  const toY = v => pad.t + h - ((v - minY) / rangeY) * h;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = DPR;
  for (let i = 0; i < 4; i++) {
    const y = pad.t + (i / 3) * h;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    const val = maxY - (i / 3) * rangeY;
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = `${9 * DPR}px SF Mono, monospace`;
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(val > 100 ? 0 : 1), pad.l - 6 * DPR, y + 3 * DPR);
  }

  // Target line
  if (opts.target) {
    const ty = toY(opts.target);
    ctx.strokeStyle = 'rgba(255,51,85,0.5)';
    ctx.setLineDash([6 * DPR, 4 * DPR]);
    ctx.lineWidth = DPR;
    ctx.beginPath(); ctx.moveTo(pad.l, ty); ctx.lineTo(W - pad.r, ty); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ff3355';
    ctx.font = `${9 * DPR}px SF Mono, monospace`;
    ctx.textAlign = 'left';
    ctx.fillText('WR ' + opts.target.toFixed(1), W - pad.r - 40 * DPR, ty - 4 * DPR);
  }

  // Gradient fill
  const grad = ctx.createLinearGradient(0, pad.t, 0, H - pad.b);
  const rgb = color === '#00ff88' ? '0,255,136' : color === '#4488ff' ? '68,136,255' : color === '#00ddff' ? '0,221,255' : '170,119,255';
  grad.addColorStop(0, `rgba(${rgb},0.15)`);
  grad.addColorStop(1, `rgba(${rgb},0)`);

  ctx.beginPath();
  ctx.moveTo(toX(0), H - pad.b);
  for (let i = 0; i < sd.length; i++) ctx.lineTo(toX(i), toY(sd[i]));
  ctx.lineTo(toX(sd.length - 1), H - pad.b);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Line with glow
  ctx.shadowColor = color;
  ctx.shadowBlur = 8 * DPR;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5 * DPR;
  ctx.beginPath();
  for (let i = 0; i < sd.length; i++) {
    if (i === 0) ctx.moveTo(toX(i), toY(sd[i]));
    else ctx.lineTo(toX(i), toY(sd[i]));
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Second series
  if (opts.data2) {
    const sd2 = opts.data2.map((_, i) => {
      const start = Math.max(0, i - smooth + 1);
      const slice = opts.data2.slice(start, i + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    });
    ctx.shadowColor = opts.color2 || '#aa77ff';
    ctx.shadowBlur = 6 * DPR;
    ctx.strokeStyle = opts.color2 || '#aa77ff';
    ctx.lineWidth = 1.5 * DPR;
    ctx.beginPath();
    for (let i = 0; i < sd2.length; i++) {
      const v = pad.t + h - ((sd2[i] - minY) / rangeY) * h;
      if (i === 0) ctx.moveTo(toX(i), v); else ctx.lineTo(toX(i), v);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

function fmtTime(s) {
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return Math.floor(s / 60) + 'm ' + (s % 60).toFixed(0) + 's';
  return Math.floor(s / 3600) + 'h ' + Math.floor((s % 3600) / 60) + 'm';
}

async function refresh() {
  try {
    const [metricsRes, expRes, wrRes, framesRes] = await Promise.all([
      fetch('/api/metrics'), fetch('/api/experiments'), fetch('/api/wr'), fetch('/api/latest_frames')
    ]);
    const metrics = await metricsRes.json();
    const exps = await expRes.json();
    const wr = await wrRes.json();
    const frames = await framesRes.json();

    // Stats
    if (metrics.length) {
      const last = metrics[metrics.length - 1];
      document.getElementById('s-steps').textContent = (last.step || 0).toLocaleString();
      document.getElementById('s-episodes').textContent = (last.total_episodes || 0).toLocaleString();
      document.getElementById('s-bestlap').textContent = last.best_lap_time && last.best_lap_time < 900 ? last.best_lap_time.toFixed(1) + 's' : '--';
      document.getElementById('s-fps').textContent = (last.fps || 0).toFixed(0);
      document.getElementById('s-mem').textContent = ((last.peak_memory_mb || 0) / 1024).toFixed(1) + ' GB';
      document.getElementById('s-elapsed').textContent = fmtTime(last.elapsed_seconds || 0);
    }

    // Charts
    const rewards = metrics.map(m => m.episode_reward).filter(x => x !== undefined);
    const pLoss = metrics.map(m => m.policy_loss).filter(x => x !== undefined);
    const vLoss = metrics.map(m => m.value_loss).filter(x => x !== undefined);
    const laps = metrics.map(m => m.avg_lap_time).filter(x => x && x < 900);
    const fps = metrics.map(m => m.fps).filter(x => x !== undefined);

    if (rewards.length) drawChart('c-reward', rewards, '#00ff88', {smooth: 8});
    if (pLoss.length) drawChart('c-loss', pLoss, '#4488ff', {smooth: 5, data2: vLoss.length > 1 ? vLoss : null, color2: '#aa77ff'});
    if (laps.length) drawChart('c-lap', laps, '#00ddff', {smooth: 3, target: 11.174});
    // FPS chart removed from layout

    // Live view + frame cache for fullscreen replay
    if (frames.length) {
      window._frameMeta = frames;
      for (const f of frames) {
        if (!window._imgCache[f.url]) {
          const img = new Image();
          img.src = f.url;
          window._imgCache[f.url] = img;
        }
      }
      // Frames loaded for replay (no live-step element in grid mode)
    }

    // WR tracker
    let whtml = '';
    for (const [track, info] of Object.entries(wr)) {
      const name = track.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
      const pct = Math.min(100, Math.max(0, info.progress_pct));
      whtml += `<div class="wr-item"><div class="wr-name"><span>${name}</span><span>${info.agent_time < 900 ? info.agent_time.toFixed(1) : '--'}s / ${info.wr_time.toFixed(1)}s</span></div><div class="wr-bar"><div class="wr-fill" style="width:${pct}%"></div></div></div>`;
    }
    document.getElementById('wr-tracker').innerHTML = whtml;

    // Experiments
    let ehtml = '';
    for (const e of exps) {
      ehtml += `<tr><td>${e.commit}</td><td>${e.avg_lap_time}</td><td>${e.tracks_wr}</td><td><span class="tag ${e.status}">${e.status}</span></td><td>${e.description}</td></tr>`;
    }
    document.getElementById('exp-body').innerHTML = ehtml || '<tr><td colspan="5" style="color:rgba(255,255,255,0.3);text-align:center;padding:20px">No experiments yet</td></tr>';

    document.getElementById('status').textContent = `${metrics.length} pts | ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    document.getElementById('status').textContent = 'error: ' + e.message;
  }
}

// Live grid state
window._frameMeta = [];
window._imgCache = {};
const N_ENVS = 8;

// Build live grid cells
const lgrid = document.getElementById('live-grid');
lgrid.innerHTML = Array.from({length: N_ENVS}, (_, i) =>
  `<div class="live-cell"><div class="env-id">#${i}</div><img id="lf-${i}" src="" alt=""></div>`
).join('');

// Poll all 8 live frames at ~5 FPS
setInterval(() => {
  const t = Date.now();
  for (let i = 0; i < N_ENVS; i++) {
    const img = document.getElementById('lf-' + i);
    if (img) img.src = '/api/live/' + i + '?t=' + t;
  }
}, 200);

// Fullscreen video viewer
window._fsIdx = 0;
window._fsCellIdx = 0;

function openFullscreen(cellIdx) {
  window._fsCellIdx = cellIdx || 0;
  const frames = window._frameMeta;
  if (!frames.length) return;
  const N = 9;
  const segLen = Math.max(1, Math.floor(frames.length / N));
  window._fsIdx = Math.min(window._fsCellIdx * segLen, frames.length - 1);
  document.getElementById('fs-overlay').classList.add('show');
  renderFullscreen();
}

function closeFullscreen() {
  document.getElementById('fs-overlay').classList.remove('show');
}

function fsNav(dir) {
  window._fsIdx = Math.max(0, Math.min(window._fsIdx + dir, window._frames.length - 1));
  renderFullscreen();
}

function renderFullscreen() {
  const frames = window._frameMeta;
  if (!frames.length) return;
  const f = frames[window._fsIdx];
  const cached = window._imgCache[f.url];
  if (cached && cached.complete) {
    document.getElementById('fs-img').src = cached.src;
  }
  const step = f.name.replace('.png', '').replace(/^0+/, '') || '0';
  document.getElementById('fs-step').textContent = `Step ${step} (${window._fsIdx + 1}/${frames.length})`;
}

// Auto-advance fullscreen
setInterval(() => {
  if (document.getElementById('fs-overlay').classList.contains('show')) {
    fsNav(1);
  }
}, 200);

// Keyboard navigation
document.addEventListener('keydown', e => {
  if (document.getElementById('fs-overlay').classList.contains('show')) {
    if (e.key === 'Escape') closeFullscreen();
    if (e.key === 'ArrowLeft') fsNav(-1);
    if (e.key === 'ArrowRight') fsNav(1);
  }
});

refresh();
setInterval(refresh, 2000);
window.addEventListener('resize', refresh);
</script>
</body>
</html>"""


async def index(request):
    return web.Response(text=HTML, content_type="text/html")


async def api_metrics(request):
    """Return latest metrics from the most recent run."""
    metrics = []
    runs = sorted(RUNS_DIR.glob("*/metrics.jsonl"), key=os.path.getmtime, reverse=True)
    if runs:
        with open(runs[0]) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return web.json_response(metrics[-500:])


async def api_experiments(request):
    """Return experiment history from results.tsv."""
    experiments = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            lines = f.readlines()
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    experiments.append({
                        "commit": parts[0],
                        "avg_lap_time": parts[1],
                        "tracks_wr": parts[2],
                        "memory_gb": parts[3],
                        "status": parts[4],
                        "description": parts[5],
                    })
    return web.json_response(experiments)


async def api_wr(request):
    """Return world record comparison."""
    result = {}
    if RECORDS_FILE.exists():
        with open(RECORDS_FILE) as f:
            records = json.load(f)["tracks"]
        best_agent_time = 999.0
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split("\t")
                    if len(parts) >= 5 and parts[4] == "keep":
                        try:
                            t = float(parts[1])
                            if 0 < t < best_agent_time:
                                best_agent_time = t
                        except ValueError:
                            pass
        for track_name, info in records.items():
            wr = info["avg_lap_wr"]
            start = 120.0
            progress = max(0, (start - best_agent_time) / (start - wr) * 100)
            result[track_name] = {"wr_time": wr, "agent_time": best_agent_time, "progress_pct": progress}
    return web.json_response(result)


async def api_latest_frame(request):
    """Return the live frame from env 0."""
    return await api_live_frame(request, env_id=0)


async def api_live_frame(request, env_id=None):
    """Return live frame for a specific env."""
    if env_id is None:
        env_id = int(request.match_info.get("env_id", 0))
    runs = sorted(RUNS_DIR.glob(f"*/live_{env_id}.png"), key=os.path.getmtime, reverse=True)
    if runs:
        return web.FileResponse(runs[0], headers={"Cache-Control": "no-cache"})
    return web.Response(status=404)


async def api_latest_frames(request):
    """Return list of frame URLs from the latest run (last 16 frames)."""
    runs = sorted(RUNS_DIR.glob("*/frames"), key=os.path.getmtime, reverse=True)
    result = []
    if runs:
        run_id = runs[0].parent.name
        frames = sorted(runs[0].glob("*.png"))
        for f in frames[-180:]:
            result.append({"url": f"/api/frame/{run_id}/{f.name}", "name": f.name, "run_id": run_id})
    return web.json_response(result)


async def api_frame(request):
    """Serve an individual frame PNG by run_id and filename."""
    run_id = request.match_info["run_id"]
    filename = request.match_info["filename"]
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        return web.Response(status=400, text="Invalid run_id")
    if "/" in filename or "\\" in filename or ".." in filename:
        return web.Response(status=400, text="Invalid filename")
    frame_path = RUNS_DIR / run_id / "frames" / filename
    if frame_path.exists() and frame_path.suffix == ".png":
        return web.FileResponse(frame_path)
    return web.Response(status=404)


CROP_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>Frame Crop Tool</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;color:#fff;font-family:'SF Mono',monospace;padding:24px}
h1{font-size:16px;letter-spacing:3px;margin-bottom:16px}
.container{display:flex;gap:24px;align-items:flex-start}
.source{position:relative;border:1px solid #333}
.source canvas{cursor:crosshair;image-rendering:pixelated}
.preview{display:flex;flex-direction:column;gap:12px}
.preview canvas{image-rendering:pixelated;border:1px solid #333}
.info{font-size:13px;line-height:1.8}
.coords{background:#111;border:1px solid #333;padding:10px 14px;border-radius:4px;font-size:14px;cursor:pointer;user-select:all;margin-top:8px;transition:border-color 0.2s}
.coords:hover{border-color:#00ddff}
.coords:active{border-color:#00ff88}
.hint{font-size:11px;color:rgba(255,255,255,0.4);margin-top:4px}
label{font-size:12px;display:flex;align-items:center;gap:8px;margin-top:8px}
input[type=range]{width:120px}
</style>
</head><body>
<h1>FRAME CROP TOOL</h1>
<div class="container">
  <div class="source">
    <canvas id="src" width="512" height="448"></canvas>
  </div>
  <div class="preview">
    <div class="info">
      <div>Original: <b>256 x 224</b></div>
      <div>Crop: <b id="crop-dims">--</b></div>
      <div>Output: <b>84 x 84 grayscale</b></div>
    </div>
    <div>Preview (84x84 B/W):</div>
    <canvas id="prev" width="168" height="168" style="width:168px;height:168px"></canvas>
    <div>Copy this config:</div>
    <div class="coords" id="coords" onclick="navigator.clipboard.writeText(this.textContent).then(()=>this.style.borderColor='#00ff88')" title="Click to copy">CROP_TOP=24, CROP_BOT=110, CROP_LEFT=0, CROP_RIGHT=256</div>
    <div class="hint">Click to copy. Paste in chat to set crop region.</div>
    <label>Frame: <input type="range" id="frame-slider" min="0" max="0" value="0" oninput="loadFrame(this.value)"> <span id="frame-num">0</span></label>
  </div>
</div>
<script>
let frames = [];
let cropY1 = 24, cropY2 = 110, cropX1 = 0, cropX2 = 256;
let dragging = false, dragStart = null;
const srcCanvas = document.getElementById('src');
const srcCtx = srcCanvas.getContext('2d');
const prevCanvas = document.getElementById('prev');
const prevCtx = prevCanvas.getContext('2d');
let currentImg = null;

async function init() {
  const res = await fetch('/api/latest_frames');
  frames = await res.json();
  document.getElementById('frame-slider').max = Math.max(0, frames.length - 1);
  if (frames.length) loadFrame(Math.floor(frames.length / 2));
}

function loadFrame(idx) {
  if (!frames[idx]) return;
  document.getElementById('frame-num').textContent = idx;
  const img = new Image();
  img.onload = () => { currentImg = img; drawSource(); updatePreview(); };
  img.src = frames[idx].url;
}

function drawSource() {
  if (!currentImg) return;
  srcCtx.clearRect(0, 0, 512, 448);
  srcCtx.drawImage(currentImg, 0, 0, 512, 448);
  // Draw crop overlay
  srcCtx.fillStyle = 'rgba(255,0,50,0.3)';
  srcCtx.fillRect(0, 0, 512, cropY1 * 2);  // top
  srcCtx.fillRect(0, cropY2 * 2, 512, 448 - cropY2 * 2);  // bottom
  srcCtx.fillRect(0, cropY1 * 2, cropX1 * 2, (cropY2 - cropY1) * 2);  // left
  srcCtx.fillRect(cropX2 * 2, cropY1 * 2, 512 - cropX2 * 2, (cropY2 - cropY1) * 2);  // right
  // Crop border
  srcCtx.strokeStyle = '#00ddff';
  srcCtx.lineWidth = 2;
  srcCtx.strokeRect(cropX1 * 2, cropY1 * 2, (cropX2 - cropX1) * 2, (cropY2 - cropY1) * 2);
}

function updatePreview() {
  if (!currentImg) return;
  const w = cropX2 - cropX1, h = cropY2 - cropY1;
  document.getElementById('crop-dims').textContent = w + ' x ' + h;
  // Draw cropped region to temp canvas, then grayscale to preview
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(currentImg, cropX1, cropY1, w, h, 0, 0, w, h);
  const imgData = tctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    const gray = d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114;
    d[i] = d[i+1] = d[i+2] = gray;
  }
  tctx.putImageData(imgData, 0, 0);
  prevCtx.clearRect(0, 0, 168, 168);
  prevCtx.imageSmoothingEnabled = false;
  prevCtx.drawImage(tmp, 0, 0, 168, 168);
  // Update coords text
  document.getElementById('coords').textContent =
    `CROP_TOP=${cropY1}, CROP_BOT=${cropY2}, CROP_LEFT=${cropX1}, CROP_RIGHT=${cropX2}`;
}

srcCanvas.addEventListener('mousedown', e => {
  const r = srcCanvas.getBoundingClientRect();
  dragStart = {x: Math.round((e.clientX - r.left) / 2), y: Math.round((e.clientY - r.top) / 2)};
  dragging = true;
});
srcCanvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  const r = srcCanvas.getBoundingClientRect();
  const x = Math.round((e.clientX - r.left) / 2);
  const y = Math.round((e.clientY - r.top) / 2);
  cropX1 = Math.max(0, Math.min(dragStart.x, x));
  cropY1 = Math.max(0, Math.min(dragStart.y, y));
  cropX2 = Math.min(256, Math.max(dragStart.x, x));
  cropY2 = Math.min(224, Math.max(dragStart.y, y));
  drawSource(); updatePreview();
});
srcCanvas.addEventListener('mouseup', () => { dragging = false; });

init();
</script>
</body></html>"""


async def crop_tool(request):
    return web.Response(text=CROP_HTML, content_type="text/html")


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/crop", crop_tool)
    app.router.add_get("/api/metrics", api_metrics)
    app.router.add_get("/api/experiments", api_experiments)
    app.router.add_get("/api/wr", api_wr)
    app.router.add_get("/api/latest_frame", api_latest_frame)
    app.router.add_get("/api/live/{env_id}", api_live_frame)
    app.router.add_get("/api/latest_frames", api_latest_frames)
    app.router.add_get("/api/frame/{run_id}/{filename}", api_frame)

    print("Dashboard running at http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080, print=None)


if __name__ == "__main__":
    main()
