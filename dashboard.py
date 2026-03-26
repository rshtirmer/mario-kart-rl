"""Standalone dashboard server -- monitors training runs and serves web UI."""

import json
import os
import glob
from pathlib import Path

from aiohttp import web

RUNS_DIR = Path(__file__).parent / "runs"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
RECORDS_FILE = Path(__file__).parent / "records.json"

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Mario Kart RL Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #e0e0e0; padding: 20px; }
h1 { color: #ff4444; margin-bottom: 10px; font-size: 24px; }
h2 { color: #44aaff; margin: 20px 0 10px; font-size: 18px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.card { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 16px; }
canvas { width: 100%; height: 200px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #222; }
th { color: #888; }
.keep { color: #44ff44; } .discard { color: #888; } .crash { color: #ff4444; }
.wr-bar { background: #222; height: 20px; border-radius: 4px; overflow: hidden; margin: 4px 0; }
.wr-fill { height: 100%; background: linear-gradient(90deg, #ff4444, #ffaa00, #44ff44); border-radius: 4px; }
.frame-viewer { text-align: center; }
.frame-viewer img { max-width: 100%; image-rendering: pixelated; border: 1px solid #333; }
#status { color: #888; font-size: 12px; margin-top: 10px; }
</style>
</head>
<body>
<h1>MARIO KART RL DASHBOARD</h1>
<div id="status">Connecting...</div>

<div class="grid">
  <div class="card">
    <h2>Training Curves</h2>
    <canvas id="rewardChart"></canvas>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="card">
    <h2>Lap Times</h2>
    <canvas id="lapChart"></canvas>
    <div id="wrTracker"></div>
  </div>
  <div class="card">
    <h2>Latest Frame</h2>
    <div class="frame-viewer"><img id="latestFrame" src="" alt="No frames yet"></div>
  </div>
  <div class="card">
    <h2>Experiment History</h2>
    <div id="experiments" style="max-height:300px;overflow-y:auto;"></div>
  </div>
</div>

<script>
function drawChart(canvasId, data, label, color) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 400;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!data.length) return;

    const padding = 40;
    const w = canvas.width - padding * 2;
    const h = canvas.height - padding * 2;

    const minY = Math.min(...data);
    const maxY = Math.max(...data);
    const rangeY = maxY - minY || 1;

    // Axes
    ctx.strokeStyle = '#333';
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '20px Courier New';
    ctx.fillText(label, padding, padding - 10);
    ctx.fillText(maxY.toFixed(1), 2, padding + 10);
    ctx.fillText(minY.toFixed(1), 2, canvas.height - padding);

    // Line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = padding + (i / (data.length - 1 || 1)) * w;
        const y = canvas.height - padding - ((data[i] - minY) / rangeY) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}

async function refresh() {
    try {
        // Metrics
        const metricsRes = await fetch('/api/metrics');
        const metrics = await metricsRes.json();

        if (metrics.length) {
            drawChart('rewardChart', metrics.map(m => m.episode_reward || 0).filter(x => x), 'Episode Reward', '#44ff44');
            drawChart('lossChart', metrics.map(m => m.policy_loss || 0).filter(x => x), 'Policy Loss', '#ff8844');
            const laps = metrics.map(m => m.avg_lap_time).filter(x => x && x < 900);
            if (laps.length) drawChart('lapChart', laps, 'Avg Lap Time (s)', '#44aaff');
        }

        // Experiments
        const expRes = await fetch('/api/experiments');
        const exps = await expRes.json();
        if (exps.length) {
            let html = '<table><tr><th>Commit</th><th>Lap Time</th><th>WR</th><th>Status</th><th>Description</th></tr>';
            for (const e of exps) {
                html += `<tr class="${e.status}"><td>${e.commit}</td><td>${e.avg_lap_time}</td><td>${e.tracks_wr}</td><td>${e.status}</td><td>${e.description}</td></tr>`;
            }
            html += '</table>';
            document.getElementById('experiments').innerHTML = html;
        }

        // WR tracker
        const wrRes = await fetch('/api/wr');
        const wr = await wrRes.json();
        let wrHtml = '';
        for (const [track, info] of Object.entries(wr)) {
            const pct = Math.min(100, Math.max(0, info.progress_pct));
            wrHtml += `<div><b>${track}</b>: ${info.agent_time.toFixed(1)}s / ${info.wr_time.toFixed(1)}s WR`;
            wrHtml += `<div class="wr-bar"><div class="wr-fill" style="width:${pct}%"></div></div></div>`;
        }
        document.getElementById('wrTracker').innerHTML = wrHtml;

        // Latest frame
        const frameRes = await fetch('/api/latest_frame');
        if (frameRes.ok) {
            const blob = await frameRes.blob();
            document.getElementById('latestFrame').src = URL.createObjectURL(blob);
        }

        document.getElementById('status').textContent = `Last update: ${new Date().toLocaleTimeString()} | ${metrics.length} data points`;
    } catch (e) {
        document.getElementById('status').textContent = 'Error: ' + e.message;
    }
}

refresh();
setInterval(refresh, 5000);
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
    return web.json_response(metrics[-500:])  # last 500 data points


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

        # Find best agent times from results.tsv
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
            # Progress: how much of the gap to WR has been closed
            # Assume starting point is 120s (very slow), WR is target
            start = 120.0
            progress = max(0, (start - best_agent_time) / (start - wr) * 100)
            result[track_name] = {
                "wr_time": wr,
                "agent_time": best_agent_time,
                "progress_pct": progress,
            }

    return web.json_response(result)


async def api_latest_frame(request):
    """Return the most recent frame PNG from any run."""
    runs = sorted(RUNS_DIR.glob("*/frames"), key=os.path.getmtime, reverse=True)
    if runs:
        frames = sorted(runs[0].glob("*.png"))
        if frames:
            return web.FileResponse(frames[-1])
    return web.Response(status=404)


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/metrics", api_metrics)
    app.router.add_get("/api/experiments", api_experiments)
    app.router.add_get("/api/wr", api_wr)
    app.router.add_get("/api/latest_frame", api_latest_frame)

    print("Dashboard running at http://localhost:8080")
    web.run_app(app, host="0.0.0.0", port=8080, print=None)


if __name__ == "__main__":
    main()
