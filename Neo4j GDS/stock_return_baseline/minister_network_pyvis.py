# -*- coding: utf-8 -*-
"""
Interactive (draggable, click-to-highlight-connections) version of the Minister <-> Ministry <->
Stock network, built with pyvis - the same library and click/highlight JS pattern app.py already
uses for the shareholder network, reused here for consistency.

Unlike visualize_minister_network.py (a static PNG for reports/README), this returns an HTML
string meant to be embedded via streamlit.components.v1.html() so the graph can be dragged,
zoomed, and clicked - clicking a node highlights only its directly-connected edges/neighbors and
shows a detail panel (dates/party for a minister, connected stocks for a ministry, etc.), entirely
client-side (no Streamlit round-trip).

Run standalone to preview in a browser:
    python minister_network_pyvis.py
"""
import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network

from build_minister_network import build_graph, parse_be_date
from ministry_stock_data import REAL_MINISTRY_INFO
from minister_network_layout import radial_layout

HERE = Path(__file__).parent
CABINET_CSV = HERE / "cabinet_history.csv"
PREVIEW_HTML = HERE / "minister_network_preview.html"

BG_COLOR = "#0F172A"
PANEL_COLOR = "#1E293B"
BORDER_COLOR = "#334155"
TEXT_COLOR = "#F1F5F9"
SUBTEXT_COLOR = "#94A3B8"
SELECTED_COLOR = "#A78BFA"

TYPE_COLOR = {"ministry": "#F59E0B", "stock": "#64748B", "minister": "#38BDF8"}
TYPE_SHAPE = {"ministry": "box", "stock": "dot", "minister": "triangle"}

POSITION_SCALE = 220  # radial_layout() coordinates are small floats - scale up to pyvis pixels


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_minister_network_html(height_px: int = 800) -> str:
    cabinet = pd.read_csv(CABINET_CSV)
    cabinet["start_date"] = cabinet["start_date"].apply(parse_be_date)
    cabinet["end_date"] = cabinet["end_date"].apply(parse_be_date)
    graph = build_graph(cabinet)

    betweenness = nx.betweenness_centrality(graph, weight="weight")
    pos = radial_layout(graph)

    net = Network(height=f"{height_px}px", width="100%", bgcolor=BG_COLOR, font_color=TEXT_COLOR,
                  directed=False, cdn_resources="in_line")

    for node, data in graph.nodes(data=True):
        node_type = data["node_type"]
        x, y = pos[node]
        bt = betweenness[node]
        base_color = TYPE_COLOR[node_type]

        if node_type == "ministry":
            label = REAL_MINISTRY_INFO[node]["label_en"]
            size = 26 + 40 * bt
            title = f"<b>{label}</b><br>Ministry<br>Betweenness: {bt:.3f}"
        elif node_type == "stock":
            label = node
            size = 10
            title = f"<b>{node}</b><br>Stock"
        else:  # minister
            label = node
            size = 14 + 55 * bt
            title = f"<b>{node}</b><br>Minister<br>Betweenness: {bt:.3f}"

        net.add_node(
            node, label=label, title=title, shape=TYPE_SHAPE[node_type],
            color={"background": rgba(base_color, 0.85), "border": base_color,
                   "highlight": {"background": SELECTED_COLOR, "border": SELECTED_COLOR}},
            size=size,
            font={"color": TEXT_COLOR, "size": 13 if node_type != "stock" else 9},
            borderWidth=2,
            x=int(x * POSITION_SCALE), y=int(y * POSITION_SCALE),
            physics=False,
        )

    for u, v, data in graph.edges(data=True):
        types = {graph.nodes[u]["node_type"], graph.nodes[v]["node_type"]}
        if types == {"ministry", "stock"}:
            color, width, title = rgba("#475569", 0.5), 1, f"correlation weight: {data['weight']:.2f}"
        else:  # minister-ministry
            years = data["weight"]
            color, width = rgba("#7DD3FC", 0.55), 2
            title = f"served {years:.1f} year(s)"
        net.add_edge(u, v, color=color, width=width, title=title)

    net.set_options("""
        const options = {
          "interaction": { "dragNodes": true, "dragView": true, "zoomView": true, "hover": true, "tooltipDelay": 150 },
          "physics": { "enabled": false },
          "nodes": { "font": { "face": "Tahoma, Arial, sans-serif" } },
          "edges": { "smooth": { "enabled": true, "type": "continuous", "roundness": 0.2 }, "selectionWidth": 3 }
        }
    """)

    # ── Detail-panel data per node ───────────────────────────────────────────
    detail_rows: dict = {}
    for node, data in graph.nodes(data=True):
        node_type = data["node_type"]
        if node_type == "minister":
            rows = cabinet.loc[cabinet["minister"] == node, ["ministry", "party", "party_role", "start_date", "end_date"]].copy()
            rows["ministry"] = rows["ministry"].map(lambda m: REAL_MINISTRY_INFO.get(m, {}).get("label_en", m))
            rows["start_date"] = rows["start_date"].astype(str)
            rows["end_date"] = rows["end_date"].astype(str)
            detail_rows[node] = {
                "title": node, "color": TYPE_COLOR["minister"],
                "subtitle": f"Served {rows['ministry'].nunique()} ministr{'y' if rows['ministry'].nunique()==1 else 'ies'}",
                "columns": ["ministry", "party", "party_role", "start_date", "end_date"],
                "rows": rows.to_dict("records"),
            }
        elif node_type == "ministry":
            neighbor_stocks = sorted(
                (nb, graph.edges[node, nb]["weight"]) for nb in graph.neighbors(node)
                if graph.nodes[nb]["node_type"] == "stock"
            )
            detail_rows[node] = {
                "title": REAL_MINISTRY_INFO[node]["label_en"], "color": TYPE_COLOR["ministry"],
                "subtitle": f"{len(neighbor_stocks)} linked stocks",
                "columns": ["ticker", "correlation_weight"],
                "rows": [{"ticker": t, "correlation_weight": round(w, 3)} for t, w in neighbor_stocks],
            }
        else:  # stock
            ministry_nb = next(nb for nb in graph.neighbors(node) if graph.nodes[nb]["node_type"] == "ministry")
            weight = graph.edges[node, ministry_nb]["weight"]
            detail_rows[node] = {
                "title": node, "color": TYPE_COLOR["stock"],
                "subtitle": f"Linked to {REAL_MINISTRY_INFO[ministry_nb]['label_en']}",
                "columns": ["ministry", "correlation_weight"],
                "rows": [{"ministry": REAL_MINISTRY_INFO[ministry_nb]["label_en"], "correlation_weight": round(weight, 3)}],
            }

    html = net.generate_html()
    details_json = json.dumps(detail_rows, ensure_ascii=False)

    container_markup = f"""
    <style>
      *, *::before, *::after {{ box-sizing: border-box; }}
      body {{ margin: 0; font-family: Tahoma, Arial, sans-serif; background: {BG_COLOR}; color: {TEXT_COLOR}; }}
      .graph-shell {{ display: grid; grid-template-columns: minmax(0, 3fr) minmax(260px, 1fr); gap: 14px; align-items: start; }}
      .graph-panel {{ min-width: 0; border-radius: 12px; overflow: hidden; }}
      #mynetwork {{ border-radius: 12px; }}
      .legend-bar {{ display: flex; gap: 18px; flex-wrap: wrap; padding: 8px 14px; background: {PANEL_COLOR};
                     border: 1px solid {BORDER_COLOR}; border-radius: 8px; margin-bottom: 10px; font-size: 12px; }}
      .leg-item {{ display: flex; align-items: center; gap: 6px; }}
      .leg-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
      .leg-box {{ width: 12px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
      .leg-tri {{ width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-bottom: 10px solid; flex-shrink: 0; }}
      .detail-panel {{ border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 16px; background: {PANEL_COLOR};
                        max-height: {height_px - 20}px; overflow: auto; }}
      .detail-panel h3 {{ margin: 0 0 4px 0; font-size: 16px; font-weight: 700; }}
      .detail-panel .sub {{ margin: 0 0 12px 0; color: {SUBTEXT_COLOR}; font-size: 12px; }}
      .detail-panel table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
      .detail-panel th {{ position: sticky; top: 0; background: {PANEL_COLOR}; border-bottom: 1px solid {BORDER_COLOR};
                           padding: 6px 4px; text-align: left; color: {SUBTEXT_COLOR}; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
      .detail-panel td {{ border-bottom: 1px solid {BORDER_COLOR}; padding: 6px 4px; vertical-align: top; }}
      .detail-empty {{ color: {SUBTEXT_COLOR}; font-size: 13px; line-height: 1.6; }}
      @media (max-width: 1000px) {{ .graph-shell {{ grid-template-columns: 1fr; }} .detail-panel {{ max-height: 380px; }} }}
    </style>
    <div class="legend-bar">
      <div class="leg-item"><div class="leg-box" style="background:{rgba(TYPE_COLOR['ministry'],0.85)};border:2px solid {TYPE_COLOR['ministry']}"></div> Ministry</div>
      <div class="leg-item"><div class="leg-tri" style="border-bottom-color:{TYPE_COLOR['minister']}"></div> Minister (size = betweenness)</div>
      <div class="leg-item"><div class="leg-dot" style="background:{TYPE_COLOR['stock']}"></div> Stock</div>
      <div class="leg-item" style="opacity:.7">Drag nodes to rearrange - click a node to highlight its connections</div>
    </div>
    <div class="graph-shell">
      <div class="graph-panel"><div id="mynetwork"></div></div>
      <div class="detail-panel" id="detail-panel">
        <h3>Node details</h3>
        <p class="detail-empty">Click any node to see its connections here.</p>
      </div>
    </div>
    """
    html, n_sub = re.subn(
        r"<body>\s*<div class=\"card\"[^>]*>\s*<div id=\"mynetwork\"[^>]*></div>\s*</div>",
        f"<body>{container_markup}",
        html, count=1, flags=re.S,
    )
    if n_sub == 0:
        raise RuntimeError("container_markup substitution didn't match pyvis's generated HTML - pyvis version may have changed its template")

    interaction_script = """
    <script type="text/javascript">
    (function() {
      if (typeof network === "undefined") return;
      const detailData = __DETAILS_JSON__;
      const detailPanel = document.getElementById("detail-panel");
      const defaultNodeStyles = {};
      const defaultEdgeStyles = {};

      function snapshotDefaults() {
        nodes.get().forEach(n => { defaultNodeStyles[n.id] = { color: n.color, size: n.size }; });
        edges.get().forEach(e => { defaultEdgeStyles[e.id] = { color: e.color, width: e.width }; });
      }

      function resetStyles() {
        nodes.update(Object.entries(defaultNodeStyles).map(([id, s]) => ({ id, color: s.color, size: s.size })));
        edges.update(Object.entries(defaultEdgeStyles).map(([id, s]) => ({ id, color: s.color, width: s.width })));
        detailPanel.innerHTML = '<h3>Node details</h3><p class="detail-empty">Click any node to see its connections here.</p>';
      }

      function renderDetails(nodeId) {
        const data = detailData[nodeId];
        if (!data || !data.rows.length) {
          detailPanel.innerHTML = `<h3 style="color:${data ? data.color : '#fff'}">${data ? data.title : nodeId}</h3><p class="detail-empty">${data ? data.subtitle : ''}</p>`;
          return;
        }
        const thead = data.columns.map(c => `<th>${c}</th>`).join("");
        const tbody = data.rows.map(row => `<tr>${data.columns.map(c => `<td>${row[c] ?? ''}</td>`).join("")}</tr>`).join("");
        detailPanel.innerHTML = `
          <h3 style="color:${data.color}">${data.title}</h3>
          <p class="sub">${data.subtitle}</p>
          <table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
      }

      function highlightNode(nodeId) {
        const connEdges = network.getConnectedEdges(nodeId);
        const connNodes = new Set(network.getConnectedNodes(nodeId));
        nodes.update(nodes.get().map(n => {
          if (n.id === nodeId) return { id: n.id, color: { background: "#A78BFA", border: "#A78BFA" }, size: Math.max((n.size||10)*1.3, 24) };
          if (connNodes.has(n.id)) return { id: n.id, color: defaultNodeStyles[n.id].color, size: defaultNodeStyles[n.id].size };
          return { id: n.id, color: "rgba(100,116,139,0.12)", size: defaultNodeStyles[n.id].size };
        }));
        edges.update(edges.get().map(e => {
          if (connEdges.includes(e.id)) return { id: e.id, color: "#A78BFA", width: 3.5 };
          return { id: e.id, color: "rgba(100,116,139,0.06)", width: 1 };
        }));
        renderDetails(nodeId);
      }

      snapshotDefaults();
      network.on("click", params => {
        if (params.nodes && params.nodes.length) highlightNode(params.nodes[0]);
        else resetStyles();
      });
    })();
    </script>
    """
    interaction_script = interaction_script.replace("__DETAILS_JSON__", details_json)
    return html.replace("</body>", interaction_script + "</body>")


def main() -> None:
    html = build_minister_network_html()
    PREVIEW_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {PREVIEW_HTML} - open it in a browser to preview")


if __name__ == "__main__":
    main()
