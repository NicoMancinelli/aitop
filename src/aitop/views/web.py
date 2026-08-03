"""Embedded live web dashboard for `aitop serve`.

A single self-contained HTML page — no build step, no CDN fonts. Polls
`/api/snapshot` on an interval (and upgrades to EventSource `/api/stream`
when available). Pure presentation of the snapshot contract.
"""

from __future__ import annotations

from aitop.version import __version__

DASHBOARD_HTML = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>aitop</title>
<style>
  :root {{
    --bg: #0e1116;
    --panel: #161b22;
    --line: #30363d;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #3fb950;
    --warn: #d29922;
    --hot: #f85149;
    --cyan: #39d2c0;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 1.25rem 1.5rem 2rem;
    font: 14px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background:
      radial-gradient(1200px 600px at 10% -10%, #1a2332 0%, transparent 55%),
      radial-gradient(900px 500px at 110% 10%, #152018 0%, transparent 50%),
      var(--bg);
    color: var(--text); min-height: 100vh;
  }}
  header {{
    display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;
    margin-bottom: 1.25rem; border-bottom: 1px solid var(--line); padding-bottom: .75rem;
  }}
  h1 {{ margin: 0; font-size: 1.4rem; letter-spacing: .04em; color: var(--cyan); }}
  .sub {{ color: var(--muted); }}
  .grid {{
    display: grid; gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    margin-bottom: 1.25rem;
  }}
  .card {{
    background: color-mix(in srgb, var(--panel) 92%, transparent);
    border: 1px solid var(--line); border-radius: 6px; padding: .9rem 1rem;
  }}
  .card h2 {{
    margin: 0 0 .55rem; font-size: .72rem; text-transform: uppercase;
    letter-spacing: .08em; color: var(--cyan);
  }}
  .bar {{
    height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; margin: .4rem 0;
  }}
  .bar > i {{
    display: block; height: 100%; background: var(--accent); width: 0%;
    transition: width .35s ease, background .35s ease;
  }}
  .bar.hot > i {{ background: var(--hot); }}
  .bar.warn > i {{ background: var(--warn); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: .25rem; }}
  th, td {{ text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--line); }}
  th {{ color: var(--muted); font-weight: 600; font-size: .72rem; letter-spacing: .06em; }}
  .dot {{ display: inline-block; width: .65rem; height: .65rem; border-radius: 50%; margin-right: .4rem; }}
  .on {{ background: var(--accent); }}
  .deg {{ background: var(--warn); }}
  .off {{ background: var(--muted); }}
  footer {{ margin-top: 1.5rem; color: var(--muted); font-size: .75rem; }}
  a {{ color: var(--cyan); }}
</style>
</head>
<body>
<header>
  <h1>aitop</h1>
  <span class="sub" id="subtitle">connecting…</span>
</header>
<section class="grid" id="metrics"></section>
<section class="card" style="margin-bottom:1rem">
  <h2>Runtimes</h2>
  <table>
    <thead><tr><th></th><th>Name</th><th>Endpoint</th><th>Version</th><th>Models</th><th>Resident</th><th>tok/s</th></tr></thead>
    <tbody id="engines"></tbody>
  </table>
</section>
<section class="card">
  <h2>Resident models</h2>
  <table>
    <thead><tr><th>Model</th><th>Engine</th><th>Size</th><th>GPU</th><th>Context</th></tr></thead>
    <tbody id="loaded"></tbody>
  </table>
</section>
<footer>
  aitop {__version__} ·
  <a href="/api/snapshot">/api/snapshot</a> ·
  <a href="/api/stream">/api/stream</a> ·
  <a href="/metrics">/metrics</a>
</footer>
<script>
const $ = (id) => document.getElementById(id);
const fmtBytes = (n) => {{
  if (n == null) return "—";
  const u = ["B","KB","MB","GB","TB"];
  let v = Number(n), i = 0;
  while (v >= 1024 && i < u.length - 1) {{ v /= 1024; i++; }}
  return (i === 0 ? v.toFixed(0) : v.toFixed(1)) + " " + u[i];
}};
const pct = (n) => n == null ? "—" : Math.round(n) + "%";
const heat = (f) => f == null ? "" : f >= 0.85 ? "hot" : f >= 0.6 ? "warn" : "";
const stateDot = (s) => s === "online" ? "on" : s === "degraded" ? "deg" : "off";

function meter(label, value, fraction, extra) {{
  const f = fraction == null ? 0 : Math.max(0, Math.min(1, fraction));
  return `<div class="card"><h2>${{label}}</h2>
    <div>${{value}}</div>
    <div class="bar ${{heat(f)}}"><i style="width:${{(f*100).toFixed(1)}}%"></i></div>
    <div class="sub">${{extra || ""}}</div></div>`;
}}

function render(snap) {{
  const hw = snap.hardware || {{}};
  const cpu = hw.cpu || {{}};
  const mem = hw.memory || {{}};
  const gpus = hw.gpus || [];
  const host = (hw.host && hw.host.hostname) || snap.node || "local";
  const online = (snap.engines || []).filter(e => e.state === "online").length;
  $("subtitle").textContent = `${{host}} · ${{online}} runtime(s) · ${{(snap.duration_ms||0).toFixed(0)}} ms · ${{new Date(snap.collected_at).toLocaleTimeString()}}`;

  let gpuHtml = "";
  if (!gpus.length) {{
    gpuHtml = meter("GPU", "none detected", 0, "");
  }} else {{
    const g = gpus[0];
    const uf = (g.utilization_percent || 0) / 100;
    const vram = g.vram_total_bytes
      ? `${{fmtBytes(g.vram_used_bytes)}} / ${{fmtBytes(g.vram_total_bytes)}}`
      : "";
    gpuHtml = meter("GPU", g.name || "GPU", uf, `${{pct(g.utilization_percent)}} · ${{vram}}`);
  }}

  $("metrics").innerHTML =
    meter("CPU", cpu.model || "CPU", (cpu.load_percent||0)/100, pct(cpu.load_percent)) +
    meter("Memory", `${{fmtBytes(mem.used_bytes)}} / ${{fmtBytes(mem.total_bytes)}}`,
      (mem.total_bytes ? mem.used_bytes/mem.total_bytes : 0),
      pct(mem.total_bytes ? mem.used_bytes/mem.total_bytes*100 : null) + (mem.unified ? " · unified" : "")) +
    gpuHtml;

  $("engines").innerHTML = (snap.engines || []).map(e => {{
    const tps = e.stats && e.stats.tokens_per_second != null ? e.stats.tokens_per_second.toFixed(1) : "—";
    const bind = e.binding ? `${{e.binding.host}}:${{e.binding.port}}` : "—";
    const res = (e.loaded && e.loaded.length)
      ? fmtBytes((e.loaded||[]).reduce((a,m)=>a+(m.size_bytes||0),0)) : "—";
    return `<tr>
      <td><span class="dot ${{stateDot(e.state)}}"></span></td>
      <td>${{e.name}}</td><td>${{bind}}</td><td>${{e.version||"—"}}</td>
      <td>${{(e.models||[]).length}}</td><td>${{res}}</td><td>${{tps}}</td></tr>`;
  }}).join("") || `<tr><td colspan="7" class="sub">no engines</td></tr>`;

  const loaded = [];
  for (const e of (snap.engines || [])) for (const m of (e.loaded || [])) loaded.push({{...m, engine: e.kind}});
  $("loaded").innerHTML = loaded.map(m => {{
    const gpu = m.size_bytes && m.vram_bytes != null ? pct(m.vram_bytes/m.size_bytes*100) : "—";
    const ctx = m.context_length ? `${{m.context_used ?? "—"}}/${{m.context_length}}` : "—";
    return `<tr><td>${{m.name}}</td><td>${{m.engine}}</td><td>${{fmtBytes(m.size_bytes)}}</td>
      <td>${{gpu}}</td><td>${{ctx}}</td></tr>`;
  }}).join("") || `<tr><td colspan="5" class="sub">no resident models</td></tr>`;
}}

async function pollOnce() {{
  const r = await fetch("/api/snapshot");
  if (!r.ok) throw new Error("snapshot " + r.status);
  render(await r.json());
}}

function start() {{
  if (window.EventSource) {{
    const es = new EventSource("/api/stream");
    es.onmessage = (ev) => {{ try {{ render(JSON.parse(ev.data)); }} catch (_) {{}} }};
    es.onerror = () => {{ es.close(); setInterval(() => pollOnce().catch(()=>{{}}), 2000); pollOnce(); }};
  }} else {{
    pollOnce();
    setInterval(() => pollOnce().catch(()=>{{}}), 2000);
  }}
}}
start();
</script>
</body>
</html>
"""


def render_dashboard() -> bytes:
    return DASHBOARD_HTML.encode("utf-8")
