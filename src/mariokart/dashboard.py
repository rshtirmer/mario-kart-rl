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

.video-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:3px;height:100%}
.vcell{position:relative;background:#000;border:1px solid var(--border);overflow:hidden;cursor:pointer;aspect-ratio:256/224}
.vcell img{width:100%;height:100%;object-fit:cover;image-rendering:pixelated}
.vcell .vlabel{position:absolute;bottom:0;left:0;right:0;font-size:8px;font-family:var(--mono);color:#fff;background:rgba(0,0,0,0.7);padding:1px 4px;text-align:center}

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
  <div class="panel" style="grid-row:1/3">
    <div class="panel-title">Training Replay Grid</div>
    <div class="video-grid" id="vgrid"></div>
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

    // Video grid - split frames across 9 cells, each plays its segment
    if (frames.length) {
      window._frames = frames;
      const grid = document.getElementById('vgrid');
      const N = 9;
      // Build cells if needed
      if (!grid.children.length) {
        grid.innerHTML = Array.from({length: N}, (_, i) =>
          `<div class="vcell" onclick="openFullscreen(${i})"><img id="vc-${i}" src=""><div class="vlabel" id="vl-${i}">--</div></div>`
        ).join('');
      }
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

// Video grid state
window._frames = [];
window._gridIdxs = new Array(9).fill(0); // each cell's current frame index

// Animate grid - each cell plays a different segment of training at staggered offsets
setInterval(() => {
  const frames = window._frames;
  if (!frames.length) return;
  const N = 9;
  const segLen = Math.max(1, Math.floor(frames.length / N));

  for (let i = 0; i < N; i++) {
    const start = Math.min(i * segLen, frames.length - 1);
    const end = Math.min(start + segLen, frames.length);
    window._gridIdxs[i] = (window._gridIdxs[i] + 1) % (end - start);
    const fIdx = start + window._gridIdxs[i];
    const f = frames[fIdx];
    if (!f) continue;
    const img = document.getElementById('vc-' + i);
    const lbl = document.getElementById('vl-' + i);
    if (img) img.src = f.url;
    if (lbl) {
      const step = f.name.replace('.png', '').replace(/^0+/, '') || '0';
      lbl.textContent = 'step ' + step;
    }
  }
}, 200); // 5 FPS per cell

// Fullscreen video viewer
window._fsIdx = 0;
window._fsCellIdx = 0;

function openFullscreen(cellIdx) {
  window._fsCellIdx = cellIdx || 0;
  const frames = window._frames;
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
  const frames = window._frames;
  if (!frames.length) return;
  const f = frames[window._fsIdx];
  document.getElementById('fs-img').src = f.url + '?t=' + Date.now();
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
    """Return the most recent frame PNG from any run."""
    runs = sorted(RUNS_DIR.glob("*/frames"), key=os.path.getmtime, reverse=True)
    if runs:
        frames = sorted(runs[0].glob("*.png"))
        if frames:
            return web.FileResponse(frames[-1])
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


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/metrics", api_metrics)
    app.router.add_get("/api/experiments", api_experiments)
    app.router.add_get("/api/wr", api_wr)
    app.router.add_get("/api/latest_frame", api_latest_frame)
    app.router.add_get("/api/latest_frames", api_latest_frames)
    app.router.add_get("/api/frame/{run_id}/{filename}", api_frame)

    print("Dashboard running at http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080, print=None)


if __name__ == "__main__":
    main()
