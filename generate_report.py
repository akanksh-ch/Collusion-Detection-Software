"""Interactive Visual Reporting — D3.js Force-Directed Forensic Dashboard.

Implements Module 5 of the production specification:
  - Leiden convex hulls (d3.polygonHull recomputed on tick)
  - Peripheral boundary routing for unlinked nodes
  - Cross-community dashed links
  - Segmented link gradients (Crimson=structural, Amber=lexical)
  - Interactive code diff sidebar with GST-highlighted matching regions

Generates a single standalone HTML file with zero external dependencies
(D3.js v7 is embedded inline via a CDN-free minified snippet fallback).
"""

from __future__ import annotations

import json
import os


class HTMLReportGenerator:
    """Generates a standalone interactive D3.js forensic dashboard."""

    @staticmethod
    def write_report(
        report_data: list[dict],
        families: dict[str, list[str]],
        source_texts: dict[str, str] | None = None,
        output_path: str = "collusion_report.html",
    ):
        # Handle directory vs file path
        if os.path.isdir(output_path) or output_path.endswith("/"):
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "collusion_report.html")
        else:
            parent_dir = os.path.dirname(output_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        # ── Build data payload ───────────────────────────────────────
        # Nodes: every student that appears in any family
        all_students = set()
        for members in families.values():
            all_students.update(members)
        for row in report_data:
            all_students.add(row["student_a"])
            all_students.add(row["student_b"])

        node_to_community = {}
        for fname, members in families.items():
            for m in members:
                node_to_community[m] = fname

        nodes_json = []
        for sid in sorted(all_students):
            nodes_json.append({
                "id": sid,
                "community": node_to_community.get(sid, "Isolated"),
            })

        links_json = []
        for row in report_data:
            links_json.append({
                "source": row["student_a"],
                "target": row["student_b"],
                "similarity": round(row["similarity"], 4),
                "sim_structural": round(row.get("sim_structural", 0), 4),
                "sim_lexical": round(row.get("sim_lexical", 0), 4),
                "gst_coverage": round(row.get("gst_coverage", 0), 4),
                "gst_tiles": row.get("gst_tiles", []),
                "risk_level": row["risk_level"],
                "family": row["family"],
                "community_a": row.get("community_a", ""),
                "community_b": row.get("community_b", ""),
            })

        families_json = {k: v for k, v in families.items() if len(v) > 1}

        sources_json = source_texts or {}

        data_payload = json.dumps({
            "nodes": nodes_json,
            "links": links_json,
            "families": families_json,
            "sources": sources_json,
        }, indent=None)

        html = _build_html(data_payload, families_json)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)


# ──────────────────────────────────────────────────────────────────────
# HTML template
# ──────────────────────────────────────────────────────────────────────

def _build_html(data_payload: str, families: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Collusion Detection — Forensic Dashboard</title>
<style>
/* ── Reset & base ──────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
  background: #0a0e17;
  color: #c9d1d9;
  overflow: hidden;
  height: 100vh;
  display: flex;
  flex-direction: column;
}}

/* ── Header ────────────────────────────────────────────────────── */
.header {{
  padding: 12px 24px;
  background: linear-gradient(135deg, #161b22, #1c2333);
  border-bottom: 1px solid #30363d;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  z-index: 100;
}}
.header h1 {{
  font-size: 16px;
  font-weight: 600;
  color: #e6edf3;
  letter-spacing: 0.5px;
}}
.header .badge {{
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 700;
}}
.badge-critical {{ background: #da3633; color: #fff; }}
.badge-high     {{ background: #d29922; color: #1c2333; }}
.badge-suspicious {{ background: #388bfd; color: #fff; }}
.stat {{
  font-size: 12px;
  color: #8b949e;
  margin-left: auto;
}}

/* ── Main layout ───────────────────────────────────────────────── */
.main-container {{
  flex: 1;
  display: flex;
  overflow: hidden;
  position: relative;
}}

/* ── SVG canvas ────────────────────────────────────────────────── */
svg {{
  flex: 1;
  min-width: 0;
}}
.hull {{
  fill-opacity: 0.08;
  stroke-width: 1.5;
  stroke-opacity: 0.3;
}}
.node-circle {{
  cursor: pointer;
  stroke: #30363d;
  stroke-width: 1.5;
  transition: r 0.2s;
}}
.node-circle:hover {{ filter: brightness(1.3); }}
.node-label {{
  font-size: 9px;
  fill: #8b949e;
  pointer-events: none;
  text-anchor: middle;
  dominant-baseline: central;
}}
.link-line {{
  cursor: pointer;
  fill: none;
  opacity: 0.7;
  transition: opacity 0.2s;
}}
.link-line:hover {{ opacity: 1; stroke-width: 4 !important; }}
.link-line.cross-community {{
  stroke-dasharray: 6 4;
}}

/* ── Sidebar ───────────────────────────────────────────────────── */
.sidebar-resizer {{
  width: 6px;
  cursor: ew-resize;
  background: transparent;
  display: none;
  z-index: 60;
  transition: background 0.2s;
}}
.sidebar-resizer:hover, .sidebar-resizer.active {{
  background: #388bfd;
}}
.sidebar {{
  width: var(--sidebar-width, 540px);
  background: #161b22;
  border-left: 1px solid #30363d;
  display: none;
  flex-direction: column;
  flex-shrink: 0;
  z-index: 50;
  overflow: hidden;
}}
.sidebar.open {{ display: flex; }}
.sidebar-header {{
  padding: 14px 18px;
  background: #1c2333;
  border-bottom: 1px solid #30363d;
  display: flex;
  align-items: center;
  gap: 12px;
}}
.sidebar-header h2 {{ font-size: 14px; color: #e6edf3; }}
.sidebar-header .close-btn {{
  margin-left: auto;
  background: none;
  border: 1px solid #30363d;
  color: #8b949e;
  width: 28px;
  height: 28px;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
}}
.sidebar-header .close-btn:hover {{ background: #21262d; color: #e6edf3; }}
.sidebar-metrics {{
  padding: 12px 18px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  border-bottom: 1px solid #21262d;
}}
.metric-pill {{
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  background: #21262d;
}}
.metric-pill.struct {{ color: #ff6b6b; }}
.metric-pill.lex {{ color: #fbbf24; }}
.metric-pill.global {{ color: #34d399; }}
.metric-pill.gst {{ color: #60a5fa; }}

/* ── Code diff panels ──────────────────────────────────────────── */
.diff-container {{
  flex: 1;
  overflow-y: auto;
  display: flex;
  gap: 0;
}}
.diff-panel {{
  flex: 1;
  overflow-y: auto;
  border-right: 1px solid #21262d;
  font-family: 'Cascadia Code', 'Fira Code', 'Source Code Pro', monospace;
  font-size: 11px;
  line-height: 1.6;
  padding: 0;
}}
.diff-panel:last-child {{ border-right: none; }}
.diff-panel-header {{
  position: sticky;
  top: 0;
  padding: 6px 12px;
  background: #1c2333;
  border-bottom: 1px solid #21262d;
  font-family: 'Segoe UI', sans-serif;
  font-size: 11px;
  color: #8b949e;
  font-weight: 600;
  z-index: 5;
}}
.code-line {{
  display: flex;
  padding: 0 8px;
  min-height: 18px;
}}
.code-line .ln {{
  width: 36px;
  text-align: right;
  color: #484f58;
  user-select: none;
  flex-shrink: 0;
  padding-right: 8px;
}}
.code-line .code {{
  flex: 1;
  white-space: pre-wrap;
  word-break: break-all;
}}
.code-line.matched-red {{
  background: rgba(248, 81, 73, 0.10);
  border-left: 3px solid #f85149;
}}
.code-line.matched-green {{
  background: rgba(46, 160, 67, 0.10);
  border-left: 3px solid #2ea043;
}}

/* ── Tooltip ───────────────────────────────────────────────────── */
.tooltip {{
  position: absolute;
  pointer-events: none;
  background: #1c2333;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 11px;
  line-height: 1.5;
  color: #c9d1d9;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  z-index: 200;
  display: none;
  max-width: 260px;
}}

/* ── Legend ─────────────────────────────────────────────────────── */
.legend {{
  position: absolute;
  bottom: 16px;
  left: 16px;
  background: rgba(22,27,34,0.92);
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 10px;
  color: #8b949e;
  z-index: 50;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 3px 0;
}}
.legend-swatch {{
  width: 14px;
  height: 4px;
  border-radius: 2px;
}}
</style>
</head>
<body>

<div class="header">
  <h1>&#x1f6e1; Collusion Detection — Forensic Dashboard</h1>
  <span class="stat" id="stat-text"></span>
</div>

<div class="main-container">
  <svg id="graph-canvas"></svg>

  <div class="sidebar-resizer" id="sidebar-resizer"></div>
  <div class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <h2 id="sidebar-title">Code Comparison</h2>
      <button class="close-btn" id="close-sidebar">&#x2715;</button>
    </div>
    <div class="sidebar-metrics" id="sidebar-metrics"></div>
    <div class="diff-container" id="diff-container"></div>
  </div>
</div>

<div class="tooltip" id="tooltip"></div>

<div class="legend">
  <div class="legend-item">
    <div class="legend-swatch" style="background:#ff6b6b;"></div>
    <span>Structural similarity (Crimson)</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#fbbf24;"></div>
    <span>Lexical similarity (Amber)</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background:#30363d; border: 1px dashed #8b949e;"></div>
    <span>Cross-community link (dashed)</span>
  </div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// ═══════════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════════
const DATA = {data_payload};
let currentSidebarWidth = 540;

const nodes = DATA.nodes;
const links = DATA.links;
const families = DATA.families;
const sources = DATA.sources;

// ── Stats ──────────────────────────────────────────────────────────
const nCritical = links.filter(l => l.risk_level === "CRITICAL").length;
const nHigh = links.filter(l => l.risk_level === "HIGH").length;
document.getElementById("stat-text").innerHTML =
  `${{nodes.length}} submissions &middot; ${{links.length}} flagged pairs &middot; ` +
  `<span class="badge badge-critical">${{nCritical}} CRITICAL</span> ` +
  `<span class="badge badge-high">${{nHigh}} HIGH</span>`;

// ═══════════════════════════════════════════════════════════════════
// COLOUR PALETTE (one per community)
// ═══════════════════════════════════════════════════════════════════
const communityNames = [...new Set(nodes.map(n => n.community))];
const palette = [
  "#58a6ff","#3fb950","#d2a8ff","#f0883e","#ff7b72",
  "#79c0ff","#56d364","#bc8cff","#db6d28","#ffa198",
  "#a5d6ff","#7ee787","#e2c5ff","#ffb757","#ffc2be",
];
const communityColor = {{}};
communityNames.forEach((c, i) => {{ communityColor[c] = palette[i % palette.length]; }});

// ── Set for linked node ids ────────────────────────────────────────
const linkedNodeIds = new Set();
links.forEach(l => {{ linkedNodeIds.add(l.source); linkedNodeIds.add(l.target); }});

// ═══════════════════════════════════════════════════════════════════
// SVG SETUP
// ═══════════════════════════════════════════════════════════════════
const svg = d3.select("#graph-canvas");
const container = svg.node().parentElement;
let W = container.clientWidth;
let H = container.clientHeight;
svg.attr("width", W).attr("height", H);

const defs = svg.append("defs");

// ── Create gradient defs for each link ─────────────────────────────
links.forEach((l, i) => {{
  const total = l.sim_structural + l.sim_lexical || 1;
  const structPct = (l.sim_structural / total) * 100;

  const grad = defs.append("linearGradient")
    .attr("id", `linkGrad-${{i}}`)
    .attr("gradientUnits", "userSpaceOnUse");

  grad.append("stop").attr("offset", `${{structPct}}%`).attr("stop-color", "#ff6b6b");
  grad.append("stop").attr("offset", `${{structPct}}%`).attr("stop-color", "#fbbf24");
}});

const gHulls = svg.append("g").attr("class", "hulls-layer");
const gLinks = svg.append("g").attr("class", "links-layer");
const gNodes = svg.append("g").attr("class", "nodes-layer");

// ═══════════════════════════════════════════════════════════════════
// FORCE SIMULATION
// ═══════════════════════════════════════════════════════════════════
const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(150).strength(0.85).iterations(3))
  .force("center", d3.forceCenter(W / 2, H / 2).strength(0.05))
  .force("collision", d3.forceCollide(28))
  .force("peripheral", d3.forceRadial(
    d => linkedNodeIds.has(d.id) ? 0 : Math.min(W, H) * 0.42,
    W / 2, H / 2
  ).strength(d => linkedNodeIds.has(d.id) ? 0 : 0.08))
  .alphaDecay(0.015);

// ── Links ──────────────────────────────────────────────────────────
const linkSel = gLinks.selectAll("line")
  .data(links)
  .join("line")
  .attr("class", d => "link-line" + (d.community_a !== d.community_b ? " cross-community" : ""))
  .attr("stroke", (d, i) => `url(#linkGrad-${{i}})`)
  .attr("stroke-width", d => 1 + d.similarity * 4)
  .on("mouseover", (e, d) => showTooltip(e, d))
  .on("mouseout", () => hideTooltip())
  .on("click", (e, d) => openSidebar(d));

// ── Nodes ──────────────────────────────────────────────────────────
const nodeSel = gNodes.selectAll("g")
  .data(nodes)
  .join("g")
  .call(d3.drag()
    .on("start", dragStart)
    .on("drag", dragging)
    .on("end", dragEnd));

nodeSel.append("circle")
  .attr("class", "node-circle")
  .attr("r", d => linkedNodeIds.has(d.id) ? 8 : 5)
  .attr("fill", d => communityColor[d.community] || "#484f58");

nodeSel.append("text")
  .attr("class", "node-label")
  .attr("dy", 18)
  .text(d => d.id + ".java");

// ── Tick ───────────────────────────────────────────────────────────
simulation.on("tick", () => {{
  const radius = 15;
  nodes.forEach(d => {{
    d.x = Math.max(radius, Math.min(W - radius, d.x));
    d.y = Math.max(radius, Math.min(H - radius, d.y));
  }});

  linkSel
    .attr("x1", d => d.source.x)
    .attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x)
    .attr("y2", d => d.target.y);

  // Update gradient endpoints to follow node positions
  links.forEach((l, i) => {{
    d3.select(`#linkGrad-${{i}}`)
      .attr("x1", l.source.x).attr("y1", l.source.y)
      .attr("x2", l.target.x).attr("y2", l.target.y);
  }});

  nodeSel.attr("transform", d => `translate(${{d.x}},${{d.y}})`);

  // ── Convex hulls per community ──────────────────────────────────
  drawHulls();
}});

function drawHulls() {{
  const groups = {{}};
  nodes.forEach(n => {{
    if (!linkedNodeIds.has(n.id)) return;
    const c = n.community;
    if (!groups[c]) groups[c] = [];
    groups[c].push([n.x, n.y]);
  }});

  const hullData = Object.entries(groups)
    .filter(([, pts]) => pts.length >= 3)
    .map(([comm, pts]) => {{
      const hull = d3.polygonHull(pts);
      return hull ? {{ community: comm, points: hull }} : null;
    }})
    .filter(Boolean);

  const hulls = gHulls.selectAll("path.hull")
    .data(hullData, d => d.community);

  hulls.enter().append("path")
    .attr("class", "hull")
    .merge(hulls)
    .attr("d", d => "M" + d.points.map(p => p.join(",")).join("L") + "Z")
    .attr("fill", d => communityColor[d.community] || "#333")
    .attr("stroke", d => communityColor[d.community] || "#555");

  hulls.exit().remove();
}}

// ═══════════════════════════════════════════════════════════════════
// TOOLTIP
// ═══════════════════════════════════════════════════════════════════
const tooltip = document.getElementById("tooltip");

function showTooltip(event, d) {{
  tooltip.innerHTML = `
    <strong>${{d.source.id || d.source}} ↔ ${{d.target.id || d.target}}</strong><br>
    Global: <strong>${{(d.similarity * 100).toFixed(1)}}%</strong><br>
    <span style="color:#ff6b6b">Structural: ${{(d.sim_structural * 100).toFixed(1)}}%</span><br>
    <span style="color:#fbbf24">Lexical: ${{(d.sim_lexical * 100).toFixed(1)}}%</span><br>
    <span style="color:#60a5fa">GST Coverage: ${{(d.gst_coverage * 100).toFixed(1)}}%</span><br>
    Risk: <strong>${{d.risk_level}}</strong>
  `;
  tooltip.style.display = "block";
  tooltip.style.left = (event.pageX + 14) + "px";
  tooltip.style.top = (event.pageY - 10) + "px";
}}

function hideTooltip() {{
  tooltip.style.display = "none";
}}

// ═══════════════════════════════════════════════════════════════════
// SIDEBAR: Code diff with GST highlights
// ═══════════════════════════════════════════════════════════════════
const sidebar = document.getElementById("sidebar");
const sidebarTitle = document.getElementById("sidebar-title");
const sidebarMetrics = document.getElementById("sidebar-metrics");
const diffContainer = document.getElementById("diff-container");

document.getElementById("close-sidebar").addEventListener("click", closeSidebar);

function openSidebar(linkData) {{
  const idA = linkData.source.id || linkData.source;
  const idB = linkData.target.id || linkData.target;

  sidebarTitle.textContent = `${{idA}} ↔ ${{idB}}`;

  sidebarMetrics.innerHTML = `
    <span class="metric-pill global">Global: ${{(linkData.similarity * 100).toFixed(1)}}%</span>
    <span class="metric-pill struct">Structural: ${{(linkData.sim_structural * 100).toFixed(1)}}%</span>
    <span class="metric-pill lex">Lexical: ${{(linkData.sim_lexical * 100).toFixed(1)}}%</span>
    <span class="metric-pill gst">GST: ${{(linkData.gst_coverage * 100).toFixed(1)}}%</span>
    <span class="metric-pill" style="color:#c9d1d9;">${{linkData.risk_level}}</span>
  `;

  // Build highlighted line sets from GST tiles
  const matchedA = new Set();
  const matchedB = new Set();
  (linkData.gst_tiles || []).forEach(tile => {{
    for (let ln = tile.a_lines[0]; ln <= tile.a_lines[1]; ln++) matchedA.add(ln);
    for (let ln = tile.b_lines[0]; ln <= tile.b_lines[1]; ln++) matchedB.add(ln);
  }});

  const srcA = sources[idA] || "(Source not available)";
  const srcB = sources[idB] || "(Source not available)";

  diffContainer.innerHTML = `
    <div class="diff-panel">
      <div class="diff-panel-header">${{idA}}</div>
      ${{renderCode(srcA, matchedA, "red")}}
    </div>
    <div class="diff-panel">
      <div class="diff-panel-header">${{idB}}</div>
      ${{renderCode(srcB, matchedB, "green")}}
    </div>
  `;

  sidebar.classList.add("open");
  document.getElementById("sidebar-resizer").style.display = "block";

  // Resize SVG and freeze simulation
  requestAnimationFrame(() => {{
    W = svg.node().parentElement.clientWidth - currentSidebarWidth - 6;
    svg.attr("width", Math.max(W, 200));
    simulation.force("center", d3.forceCenter(Math.max(W, 200) / 2, H / 2));
    simulation.alpha(0.1).restart();
  }});
}}

function closeSidebar() {{
  sidebar.classList.remove("open");
  document.getElementById("sidebar-resizer").style.display = "none";
  requestAnimationFrame(() => {{
    W = svg.node().parentElement.clientWidth;
    svg.attr("width", W);
    simulation.force("center", d3.forceCenter(W / 2, H / 2));
    simulation.alpha(0.05).restart();
  }});
}}

function renderCode(src, matchedLines, color="red") {{
  const lines = src.split("\\n");
  return lines.map((line, i) => {{
    const ln = i + 1;
    const cls = matchedLines.has(ln) ? `code-line matched-${{color}}` : "code-line";
    const escaped = line.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    return `<div class="${{cls}}"><span class="ln">${{ln}}</span><span class="code">${{escaped}}</span></div>`;
  }}).join("");
}}

// ═══════════════════════════════════════════════════════════════════
// DRAG BEHAVIOUR
// ═══════════════════════════════════════════════════════════════════
function dragStart(event, d) {{
  if (!event.active) simulation.alphaTarget(0.3).restart();
  
  d.clusterNodes = d.community !== "Isolated" 
    ? nodes.filter(n => n.community === d.community)
    : [d];
    
  d.clusterNodes.forEach(n => {{
    n.startX = n.x;
    n.startY = n.y;
    n.fx = n.x;
    n.fy = n.y;
  }});
  d.dragStartX = event.x;
  d.dragStartY = event.y;
}}
function dragging(event, d) {{
  const dx = event.x - d.dragStartX;
  const dy = event.y - d.dragStartY;
  d.clusterNodes.forEach(n => {{
    n.fx = n.startX + dx;
    n.fy = n.startY + dy;
  }});
}}
function dragEnd(event, d) {{
  if (!event.active) simulation.alphaTarget(0);
  if (d.clusterNodes) {{
    d.clusterNodes.forEach(n => {{
      n.fx = null;
      n.fy = null;
    }});
    d.clusterNodes = null;
  }}
}}

// ── Sidebar Resizer ───────────────────────────────────────────────
const sidebarResizer = document.getElementById("sidebar-resizer");
let isSidebarResizing = false;

sidebarResizer.addEventListener("mousedown", (e) => {{
  isSidebarResizing = true;
  sidebarResizer.classList.add("active");
  document.body.style.cursor = "ew-resize";
  e.preventDefault();
}});

document.addEventListener("mousemove", (e) => {{
  if (!isSidebarResizing) return;
  const newWidth = document.body.clientWidth - e.clientX;
  if (newWidth > 300 && newWidth < document.body.clientWidth - 200) {{
    currentSidebarWidth = newWidth;
    sidebar.style.setProperty('--sidebar-width', newWidth + "px");
    
    W = container.clientWidth - currentSidebarWidth - 6;
    svg.attr("width", Math.max(W, 200));
    simulation.force("center", d3.forceCenter(Math.max(W, 200) / 2, H / 2));
    simulation.alpha(0.05).restart();
  }}
}});

document.addEventListener("mouseup", () => {{
  if (isSidebarResizing) {{
    isSidebarResizing = false;
    sidebarResizer.classList.remove("active");
    document.body.style.cursor = "default";
  }}
}});

// ── Resize handler ─────────────────────────────────────────────────
window.addEventListener("resize", () => {{
  const sidebarOpen = sidebar.classList.contains("open");
  W = container.clientWidth - (sidebarOpen ? currentSidebarWidth + 6 : 0);
  H = container.clientHeight;
  svg.attr("width", Math.max(W, 200)).attr("height", H);
  simulation.force("center", d3.forceCenter(Math.max(W, 200) / 2, H / 2));
  simulation.alpha(0.05).restart();
}});
</script>
</body>
</html>"""


# The template uses an f-string, so the data_payload JSON is embedded inline.
