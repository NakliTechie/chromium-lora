"""Draw the README's results chart from fork/results/results.json.

Grouped bars: for each recipe, held-out questions answered with the expected
fact (of 30) with no adapter, with the facts in the prompt, and with the site's
adapter. Writes fork/results/facts-chart.svg. Standard library only.

Usage: python3 fork/tools/results_chart.py
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

RESULTS = Path(__file__).resolve().parents[1] / "results"
ARMS = [("no adapter", "#2a78d6"), ("facts in the prompt", "#eb6834"), ("site adapter", "#1baf7a")]
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b68", "#e4e4e0", "#fcfcfb"
W, H = 720, 360
LEFT, RIGHT, TOP, BOTTOM = 56, 24, 64, 64
BAR, GAP, GROUP_GAP = 56, 2, 72

data = json.loads((RESULTS / "results.json").read_text())
top = 30
plot_h = H - TOP - BOTTOM
y = lambda v: TOP + plot_h * (1 - v / top)
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" role="img" '
       'aria-label="Held-out questions answered with the expected fact, of 30, by recipe and method">',
       f'<rect width="{W}" height="{H}" rx="8" fill="{SURFACE}"/>',
       f'<text x="{LEFT}" y="26" font-size="15" font-weight="600" fill="{INK}">'
       f'Held-out questions answered with the expected fact (of 30)</text>',
       f'<text x="{LEFT}" y="46" font-size="12" fill="{MUTED}">{escape(data["model"])}, '
       f'measured {data["measured"]}</text>']
for v in (0, 10, 20, 30):
    out.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
    out.append(f'<text x="{LEFT - 8}" y="{y(v) + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{v}</text>')

group_w = len(ARMS) * BAR + (len(ARMS) - 1) * GAP
x0 = LEFT + (W - LEFT - RIGHT - 2 * group_w - GROUP_GAP) / 2
for g, recipe in enumerate(data["recipes"]):
    gx = x0 + g * (group_w + GROUP_GAP)
    for i, (arm, colour) in enumerate(ARMS):
        v = recipe["arms"][arm]
        bx, by = gx + i * (BAR + GAP), y(v)
        h = y(0) - by
        r = min(4, h)
        # Rounded data end, square at the baseline.
        out.append(f'<path d="M{bx},{y(0)} V{by + r} Q{bx},{by} {bx + r},{by} H{bx + BAR - r} '
                   f'Q{bx + BAR},{by} {bx + BAR},{by + r} V{y(0)} Z" fill="{colour}"><title>'
                   f'{escape(recipe["name"])}, {arm}: {v} of 30</title></path>')
        out.append(f'<text x="{bx + BAR / 2}" y="{by - 6}" font-size="12" font-weight="600" fill="{INK}" '
                   f'text-anchor="middle">{v}</text>')
    out.append(f'<text x="{gx + group_w / 2}" y="{y(0) + 20}" font-size="12" fill="{INK}" '
               f'text-anchor="middle">{escape(recipe["name"])}</text>')

lx = LEFT
for arm, colour in ARMS:
    out.append(f'<rect x="{lx}" y="{H - 22}" width="10" height="10" rx="2" fill="{colour}"/>')
    out.append(f'<text x="{lx + 16}" y="{H - 13}" font-size="12" fill="{MUTED}">{arm}</text>')
    lx += 16 + 8 * len(arm) + 24
out.append("</svg>")
(RESULTS / "facts-chart.svg").write_text("\n".join(out) + "\n")
print(RESULTS / "facts-chart.svg")
