"""Dependency-free, loopback-only live dashboard for a single JSONL metrics file."""

import argparse
import json
import math
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

BATCH_BYTES = 1024 * 1024


def finite_float(text):
    value = float(text)
    return value if math.isfinite(value) else None


def read_metrics(path, cursor=0, generation=None):
    """Read at most 1 MiB, advancing only past complete newline-delimited records."""
    try:
        with path.open("rb") as stream:
            info = os.fstat(stream.fileno())
            identity = f"{info.st_dev}:{info.st_ino}"
            reset = generation != identity or cursor > info.st_size
            if reset:
                cursor = 0
            stream.seek(cursor)
            data = stream.read(BATCH_BYTES)
    except FileNotFoundError:
        return {"rows": [], "cursor": 0, "generation": None, "reset": True, "missing": True}
    end = data.rfind(b"\n") + 1
    if not end and len(data) == BATCH_BYTES:
        raise ValueError("A metrics record exceeds 1 MiB; use smaller JSONL records")
    rows, skipped = [], 0
    for line in data[:end].splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_float=finite_float, parse_constant=lambda _: None)
            if not isinstance(row, dict):
                raise ValueError("Expected a JSON object")
            rows.append(row)
        except (ValueError, UnicodeError, RecursionError):
            skipped += 1
    return {
        "rows": rows,
        "cursor": cursor + end,
        "generation": identity,
        "reset": reset,
        "missing": False,
        "skipped": skipped,
        "has_more": len(data) == BATCH_BYTES and cursor + end < info.st_size,
        "modified": info.st_mtime,
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanospec · local metrics</title>
<style>
:root { color-scheme: dark; font: 15px/1.5 system-ui, sans-serif; color: #dbe5f5;
  background: #0b1020; --muted: #92a0b8; --border: #263148; }
* { box-sizing: border-box; }
body { margin: 0; background: radial-gradient(ellipse at top left, #172342, transparent 65%); }
main { max-width: 1320px; margin: auto; padding: 40px 28px; }
header, .toolbar, .card-heading, .summary { display: flex; align-items: center;
  justify-content: space-between; gap: 18px; flex-wrap: wrap; }
h1 { margin: 0; font-size: 30px; letter-spacing: -.8px; }
.brand { color: #72dcc8; font-size: 12px; letter-spacing: 2px; text-transform: uppercase; }
p { margin: 6px 0; color: var(--muted); }
#status { padding: 7px 13px; border: 1px solid var(--border); border-radius: 20px;
  font-size: 13px; color: #92a0b8; }
#status[data-state="live"] { color: #72dcc8; border-color: #2f655f; }
#status[data-state="error"] { color: #ffb09f; border-color: #7d4f46; }
.summary { margin: 26px 0; justify-content: flex-start; gap: 38px; }
.summary strong { display: block; font-size: 24px; font-variant-numeric: tabular-nums; }
.summary span, .caption { font-size: 12px; color: var(--muted); }
#file { overflow-wrap: anywhere; font: 13px/1.5 ui-monospace, monospace; }
.toolbar { margin-bottom: 20px; }
input, button { font: inherit; color: inherit; border: 1px solid var(--border);
  background: #121b2d; border-radius: 8px; padding: 9px 13px; }
input { width: min(100%, 300px); }
button { cursor: pointer; }
button:hover { border-color: #72dcc8; }
input:focus-visible, button:focus-visible { outline: 2px solid #72dcc8; outline-offset: 3px; }
#charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 350px), 1fr));
  gap: 18px; }
.card { background: #111a2bdd; border: 1px solid var(--border); border-radius: 14px;
  padding: 20px 18px 12px; min-width: 0; }
.card h2 { margin: 0; font-size: 14px; font-weight: 500; overflow-wrap: anywhere; }
.value { font-size: 25px; font-weight: 600; font-variant-numeric: tabular-nums; }
svg { display: block; width: 100%; height: auto; margin-top: 12px; overflow: visible; }
svg text { fill: var(--muted); font: 11px system-ui, sans-serif; }
.tooltip { min-height: 20px; margin-top: 3px; font: 12px ui-monospace, monospace;
  color: var(--muted); overflow-wrap: anywhere; }
#empty { padding: 52px 20px; text-align: center; border: 1px dashed var(--border);
  border-radius: 14px; color: var(--muted); }
footer { margin-top: 22px; color: var(--muted); font-size: 12px; }
@media (max-width: 550px) { main { padding: 24px 16px; } h1 { font-size: 26px; } }
</style>
</head>
<body>
<main>
<header><div><div class="brand">nanospec / metrics</div><h1>Your run, on your machine.</h1>
<p>No accounts. No uploads. Just a local file.</p></div>
<div id="status" role="status">Connecting</div></header>
<div class="summary"><div><strong id="step">—</strong><span>latest step</span></div>
<div><strong id="count">0</strong><span>records read</span></div>
<div><strong id="modified">—</strong><span>last file update</span></div></div>
<p id="file"></p>
<div class="toolbar"><input id="filter" type="search" placeholder="Filter metrics…"
 aria-label="Filter metrics"><button id="pause" type="button">Pause live refresh</button></div>
<div id="charts"></div><div id="empty">Waiting for metrics…</div>
<footer>Polls every 2 seconds · Charts keep the latest 2,000 points per metric · Hover for values
<div id="note" role="status"></div></footer>
</main>
<script>
'use strict';
const el = id => document.getElementById(id);
const series = new Map();
const colors = ['#72dcc8', '#98a9ff', '#f4c578', '#f19eb8', '#80caff'];
const priority = ['loss', 'accept_len', 'grad_norm', 'seconds', 'learning_rate', 'denominator'];
let cursor = 0, generation = null, count = 0, skipped = 0, paused = false;
let lastStep = null, missing = false, fetching = false;
const number = value => {
  if (value !== 0 && (Math.abs(value) < .0001 || Math.abs(value) >= 1e7)) {
    return value.toExponential(3);
  }
  return new Intl.NumberFormat('en', { maximumSignificantDigits: 5 }).format(value);
};
function node(tag, className, text) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (text !== undefined) result.textContent = text;
  return result;
}
function svgNode(tag, attributes, text) {
  const result = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) result.setAttribute(key, String(value));
  if (text !== undefined) result.textContent = text;
  return result;
}
function chart(key, points, color) {
  const card = node('section', 'card');
  const heading = node('div', 'card-heading');
  heading.append(node('h2', '', key.replaceAll('_', ' ')),
    node('div', 'value', number(points.at(-1)[1])));
  card.append(heading);
  const svg = svgNode('svg', { viewBox: '0 0 640 220', role: 'img',
    'aria-label': key + ' over step (or record number when step is absent)' });
  let low = Infinity, high = -Infinity, first = Infinity, last = -Infinity;
  for (const [x, y] of points) {
    low = Math.min(low, y); high = Math.max(high, y);
    first = Math.min(first, x); last = Math.max(last, x);
  }
  // Scale before subtraction to avoid overflow for very large finite values.
  const yScale = Math.max(Math.abs(low), Math.abs(high), 1e-100);
  const xScale = Math.max(Math.abs(first), Math.abs(last), 1);
  const lo = low / yScale, hi = high / yScale;
  const left = first / xScale, right = last / xScale;
  const yRange = hi - lo || Math.max(Math.abs(lo) * .1, .1);
  const yMid = (lo + hi) / 2;
  const px = x => 76 + (right === left ? .5 : (x / xScale - left) / (right - left)) * 548;
  const py = y => 180 - ((y / yScale - yMid) / yRange + .5) * 150;
  for (const [value, position] of [[high, 30], [(low / 2 + high / 2), 105], [low, 180]]) {
    svg.append(svgNode('line', { x1: 76, x2: 624, y1: position, y2: position,
      stroke: '#263148', 'stroke-dasharray': '3 5' }));
    svg.append(svgNode('text', { x: 66, y: position + 4, 'text-anchor': 'end' }, number(value)));
  }
  svg.append(svgNode('text', { x: 76, y: 209 }, number(first)),
    svgNode('text', { x: 350, y: 209, 'text-anchor': 'middle' }, 'step / record'),
    svgNode('text', { x: 624, y: 209, 'text-anchor': 'end' }, number(last)));
  const path = points.map(([x, y], index) =>
    (index ? 'L' : 'M') + px(x).toFixed(2) + ',' + py(y).toFixed(2)).join(' ');
  svg.append(svgNode('path', { d: path, stroke: color, fill: 'none', 'stroke-width': 2.5,
    'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
  const point = points.at(-1);
  svg.append(svgNode('circle', { cx: px(point[0]), cy: py(point[1]), r: 3.5, fill: color }));
  const tooltip = node('div', 'tooltip', points.length + ' points · latest at ' + number(point[0]));
  svg.addEventListener('pointermove', event => {
    const bounds = svg.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width * 640;
    let nearest = points[0];
    for (const candidate of points) {
      if (Math.abs(px(candidate[0]) - x) < Math.abs(px(nearest[0]) - x)) nearest = candidate;
    }
    tooltip.textContent = 'at ' + number(nearest[0]) + ' → ' + number(nearest[1]);
  });
  card.append(svg, tooltip);
  return card;
}
function render() {
  const query = el('filter').value.toLowerCase();
  const keys = [...series.keys()].sort((a, b) => {
    const rank = key => priority.includes(key) ? priority.indexOf(key) : priority.length;
    return rank(a) - rank(b) || a.localeCompare(b);
  });
  const cards = [];
  for (const [index, key] of keys.entries()) {
    if (key.toLowerCase().includes(query)) cards.push(chart(key, series.get(key), colors[index % 5]));
  }
  el('charts').replaceChildren(...cards);
  el('empty').hidden = cards.length > 0;
  el('empty').textContent = missing ? 'File not found yet. Start training or check the displayed path.'
    : count === 0 ? 'Waiting for complete JSONL records…'
    : 'No matching numeric metrics. Numeric top-level fields are charted automatically.';
  el('count').textContent = number(count);
  el('step').textContent = lastStep === null ? '—' : number(lastStep);
}
async function poll() {
  if (paused || fetching) return;
  fetching = true;
  try {
    const query = new URLSearchParams({ cursor: String(cursor) });
    if (generation !== null) query.set('generation', generation);
    const response = await fetch('/api/metrics?' + query, { cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'HTTP ' + response.status);
    if (data.reset) { series.clear(); count = 0; skipped = 0; lastStep = null; }
    missing = data.missing;
    cursor = data.cursor; generation = data.generation;
    skipped += data.skipped || 0;
    for (const row of data.rows) {
      count += 1;
      const step = typeof row.step === 'number' && Number.isFinite(row.step) ? row.step : count;
      lastStep = step;
      for (const [key, value] of Object.entries(row)) {
        if (key === 'step' || key === 'timestamp' || typeof value !== 'number'
            || !Number.isFinite(value)) continue;
        if (!series.has(key)) series.set(key, []);
        const points = series.get(key);
        points.push([step, value]);
        if (points.length > 2000) points.shift();
      }
    }
    el('file').textContent = data.file;
    el('modified').textContent = data.modified ? new Date(data.modified * 1000).toLocaleTimeString() : '—';
    el('status').textContent = paused ? 'Paused' : missing ? 'Waiting for file' : 'Live · local only';
    el('status').dataset.state = missing || paused ? '' : 'live';
    el('note').textContent = skipped ? skipped + ' malformed/non-object records skipped.' : '';
    if (data.reset || data.rows.length) render();
    if (data.has_more && !paused) setTimeout(poll, 50);
  } catch (error) {
    el('status').textContent = 'Read error';
    el('status').dataset.state = 'error';
    el('note').textContent = error.message;
  } finally { fetching = false; }
}
el('filter').addEventListener('input', render);
el('pause').addEventListener('click', () => {
  paused = !paused;
  el('pause').textContent = paused ? 'Resume live refresh' : 'Pause live refresh';
  el('status').textContent = paused ? 'Paused' : 'Connecting';
  el('status').dataset.state = '';
  if (!paused) poll();
});
poll();
setInterval(poll, 2000);
</script>
</body>
</html>"""


def make_server(path, port):
    path = Path(path).expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, status, payload):
            self.send(status, json.dumps(payload, allow_nan=False).encode(), "application/json")

        def do_GET(self):
            # Also reject foreign Host headers to prevent browser DNS-rebinding access.
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if port == 80:
                hosts.update(("127.0.0.1", "localhost"))
            if self.headers.get("Host") not in hosts:
                self.send_json(403, {"error": "Use the loopback dashboard URL"})
                return
            url = urlsplit(self.path)
            if url.path == "/" and not url.query:
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/metrics":
                try:
                    query = parse_qs(url.query, keep_blank_values=True, max_num_fields=2)
                    if query.keys() - {"cursor", "generation"}:
                        raise ValueError("Only cursor and generation parameters are supported")
                    if any(len(values) != 1 for values in query.values()):
                        raise ValueError("Duplicate query parameters")
                    cursor = query.get("cursor", ["0"])[0]
                    if len(cursor) > 20 or not cursor.isascii() or not cursor.isdecimal():
                        raise ValueError("cursor must be a non-negative byte offset")
                    generation = query.get("generation", [None])[0]
                    if generation is not None and len(generation) > 100:
                        raise ValueError("Invalid generation")
                    payload = read_metrics(path, int(cursor), generation)
                    self.send_json(200, {**payload, "file": str(path)})
                except ValueError as error:
                    self.send_json(400, {"error": str(error)})
                except OSError as error:
                    self.send_json(503, {"error": f"Cannot read metrics file: {error}"})
            else:
                self.send_json(404, {"error": "Not found"})

        def log_message(self, format, *args):
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_file", type=Path, help="The only JSONL file exposed by this server"
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with make_server(args.metrics_file, args.port) as server:
        print(f"Dashboard: http://127.0.0.1:{server.server_port}/", flush=True)
        print(f"Metrics: {args.metrics_file.expanduser().resolve()} (Ctrl-C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
