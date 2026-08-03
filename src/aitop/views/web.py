"""Embedded live web dashboard for `aitop serve`.

A single self-contained HTML page — no build step, no CDN fonts. Polls
`/api/snapshot` on an interval (and upgrades to EventSource `/api/stream`
when available). Includes control actions that POST to the serve API.
"""

from __future__ import annotations

from aitop.version import __version__


def render_dashboard(*, auth_required: bool = False) -> bytes:
    auth_flag = "true" if auth_required else "false"
    html = f"""\
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
    margin-bottom: 1rem;
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
  th, td {{ text-align: left; padding: .35rem .5rem; border-bottom: 1px solid var(--line); vertical-align: middle; }}
  th {{ color: var(--muted); font-weight: 600; font-size: .72rem; letter-spacing: .06em; }}
  .dot {{ display: inline-block; width: .65rem; height: .65rem; border-radius: 50%; margin-right: .4rem; }}
  .on {{ background: var(--accent); }}
  .deg {{ background: var(--warn); }}
  .off {{ background: var(--muted); }}
  button {{
    font: inherit; cursor: pointer; color: var(--text);
    background: #21262d; border: 1px solid var(--line); border-radius: 4px;
    padding: .15rem .45rem; margin: 0 .15rem .15rem 0;
  }}
  button:hover {{ border-color: var(--cyan); color: var(--cyan); }}
  button.danger:hover {{ border-color: var(--hot); color: var(--hot); }}
  #toast {{
    position: fixed; right: 1rem; bottom: 1rem; max-width: 28rem;
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px;
    padding: .65rem .9rem; color: var(--text); display: none; z-index: 20;
    box-shadow: 0 8px 24px rgba(0,0,0,.35);
  }}
  #toast.show {{ display: block; }}
  #toast.err {{ border-color: var(--warn); }}
  footer {{ margin-top: 1.5rem; color: var(--muted); font-size: .75rem; }}
  a {{ color: var(--cyan); }}
  .token-row {{ margin-left: auto; display: flex; gap: .4rem; align-items: center; }}
  .token-row input {{
    font: inherit; background: #0e1116; color: var(--text);
    border: 1px solid var(--line); border-radius: 4px; padding: .2rem .45rem; width: 10rem;
  }}
</style>
</head>
<body>
<header>
  <h1>aitop</h1>
  <span class="sub" id="subtitle">connecting…</span>
  <div class="token-row">
    <input id="token" type="password" placeholder="serve token" autocomplete="off"/>
    <button type="button" id="save-token">save</button>
  </div>
</header>
<section class="grid" id="metrics"></section>
<section class="card">
  <h2>Runtimes</h2>
  <table>
    <thead><tr><th></th><th>Name</th><th>Endpoint</th><th>Version</th><th>Models</th><th>Resident</th><th>tok/s</th><th>Actions</th></tr></thead>
    <tbody id="engines"></tbody>
  </table>
</section>
<section class="card">
  <h2>Catalog</h2>
  <table>
    <thead><tr><th>Model</th><th>Engine</th><th>Params</th><th>Quant</th><th>Size</th><th>State</th><th>Actions</th></tr></thead>
    <tbody id="catalog"></tbody>
  </table>
</section>
<section class="card">
  <h2>Resident models</h2>
  <table>
    <thead><tr><th>Model</th><th>Engine</th><th>Size</th><th>GPU</th><th>Context</th><th>Actions</th></tr></thead>
    <tbody id="loaded"></tbody>
  </table>
</section>
<div id="toast"></div>
<footer>
  aitop {__version__} ·
  <a href="/api/snapshot">/api/snapshot</a> ·
  <a href="/api/stream">/api/stream</a> ·
  <a href="/metrics">/metrics</a>
</footer>
<script>
const AUTH_HINT = {auth_flag};
const $ = (id) => document.getElementById(id);
const TOKEN_KEY = "aitop.serve.token";
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
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({{
  "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
}})[c]);

function getToken() {{
  return localStorage.getItem(TOKEN_KEY) || $("token").value || "";
}}
function authHeaders() {{
  const t = getToken();
  return t ? {{ "Authorization": "Bearer " + t, "Content-Type": "application/json" }}
           : {{ "Content-Type": "application/json" }};
}}
function toast(msg, err) {{
  const el = $("toast");
  el.textContent = msg;
  el.className = "show" + (err ? " err" : "");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.className = "", 4200);
}}

async function api(path, body) {{
  const r = await fetch(path, {{ method: "POST", headers: authHeaders(), body: JSON.stringify(body || {{}}) }});
  let data = {{}};
  try {{ data = await r.json(); }} catch (_) {{}}
  if (r.status === 401) {{
    toast("unauthorized — set serve token", true);
    return null;
  }}
  if (!r.ok || data.ok === false) {{
    toast(data.message || data.error || ("HTTP " + r.status), true);
    return null;
  }}
  toast(data.message || "ok");
  return data;
}}

function meter(label, value, fraction, extra) {{
  const f = fraction == null ? 0 : Math.max(0, Math.min(1, fraction));
  return `<div class="card" style="margin:0"><h2>${{label}}</h2>
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
    const kind = esc(e.kind);
    const actions = `
      <button data-act="start" data-kind="${{kind}}">start</button>
      <button data-act="restart" data-kind="${{kind}}">restart</button>
      <button class="danger" data-act="stop" data-kind="${{kind}}">stop</button>`;
    return `<tr>
      <td><span class="dot ${{stateDot(e.state)}}"></span></td>
      <td>${{esc(e.name)}}</td><td>${{esc(bind)}}</td><td>${{esc(e.version||"—")}}</td>
      <td>${{(e.models||[]).length}}</td><td>${{res}}</td><td>${{tps}}</td>
      <td>${{actions}}</td></tr>`;
  }}).join("") || `<tr><td colspan="8" class="sub">no engines</td></tr>`;

  const catalog = [];
  for (const e of (snap.engines || [])) {{
    const loaded = new Set((e.loaded || []).map(m => m.id));
    for (const m of (e.models || [])) {{
      catalog.push({{...m, engine: e.kind, resident: loaded.has(m.id)}});
    }}
  }}
  $("catalog").innerHTML = catalog.map(m => {{
    const state = m.resident ? "● resident" : "○ disk";
    const kind = esc(m.engine);
    const mid = esc(m.id);
    const actions = m.resident
      ? `<button class="danger" data-mact="unload" data-kind="${{kind}}" data-model="${{mid}}">unload</button>`
      : `<button data-mact="load" data-kind="${{kind}}" data-model="${{mid}}">load</button>
         <button class="danger" data-mact="delete" data-kind="${{kind}}" data-model="${{mid}}">delete</button>`;
    return `<tr>
      <td>${{esc(m.name)}}</td><td>${{kind}}</td><td>${{esc(m.parameter_size||"—")}}</td>
      <td>${{esc(m.quantization||"—")}}</td><td>${{fmtBytes(m.size_bytes)}}</td>
      <td>${{state}}</td><td>${{actions}}</td></tr>`;
  }}).join("") || `<tr><td colspan="7" class="sub">no models on disk</td></tr>`;

  const loaded = [];
  for (const e of (snap.engines || [])) for (const m of (e.loaded || [])) loaded.push({{...m, engine: e.kind}});
  $("loaded").innerHTML = loaded.map(m => {{
    const gpu = m.size_bytes && m.vram_bytes != null ? pct(m.vram_bytes/m.size_bytes*100) : "—";
    const ctx = m.context_length ? `${{m.context_used ?? "—"}}/${{m.context_length}}` : "—";
    const kind = esc(m.engine);
    const mid = esc(m.id);
    return `<tr><td>${{esc(m.name)}}</td><td>${{kind}}</td><td>${{fmtBytes(m.size_bytes)}}</td>
      <td>${{gpu}}</td><td>${{ctx}}</td>
      <td><button class="danger" data-mact="unload" data-kind="${{kind}}" data-model="${{mid}}">unload</button></td></tr>`;
  }}).join("") || `<tr><td colspan="6" class="sub">no resident models</td></tr>`;
}}

document.body.addEventListener("click", async (ev) => {{
  const btn = ev.target.closest("button");
  if (!btn) return;
  if (btn.dataset.act) {{
    const kind = btn.dataset.kind;
    const act = btn.dataset.act;
    if (act === "stop" && !confirm("Stop " + kind + "?")) return;
    if (act === "restart" && !confirm("Restart " + kind + "?")) return;
    await api(`/api/engines/${{encodeURIComponent(kind)}}/${{act}}`, {{}});
    return;
  }}
  if (btn.dataset.mact) {{
    const act = btn.dataset.mact;
    const kind = btn.dataset.kind;
    const model = btn.dataset.model;
    if (act === "delete" && !confirm("Delete " + model + " from disk?")) return;
    await api(`/api/models/${{act}}`, {{ engine: kind, model }});
  }}
}});

$("save-token").addEventListener("click", () => {{
  localStorage.setItem(TOKEN_KEY, $("token").value || "");
  toast("token saved locally");
}});
$("token").value = localStorage.getItem(TOKEN_KEY) || "";
if (AUTH_HINT) toast("this server may require a serve token for control");

async function pollOnce() {{
  const headers = {{}};
  const t = getToken();
  if (t) headers["Authorization"] = "Bearer " + t;
  const r = await fetch("/api/snapshot", {{ headers }});
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
    return html.encode("utf-8")
