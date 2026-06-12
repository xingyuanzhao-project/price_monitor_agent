import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT  = '/Users/xingyuanzhao/Documents/projects/price_monitor_agent/plots/architecture.png'
BG   = '#0d1117'

fig  = plt.figure(figsize=(22, 30))
ax   = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Palette ────────────────────────────────────────────────────────────────
P = {
    'human':      '#58a6ff',  'human_bg':   '#0c2250',
    'fe':         '#388bfd',  'fe_bg':      '#0d1f3c',
    'pe':         '#3fb950',  'pe_bg':      '#0d2818',
    'orch':       '#bc8cff',  'orch_bg':    '#170c2f',
    'tools':      '#e3b341',  'tools_bg':   '#251a00',
    'harness':    '#ff7b72',  'harness_bg': '#2e0906',
    'agent':      '#79c0ff',  'agent_bg':   '#041224',
    'llm':        '#6e7681',  'llm_bg':     '#161b22',
    'g_fe':       '#1f6feb',
    'g_pe':       '#238636',
    'g_be':       '#6e40c9',
    'subtext':    '#8b949e',
    'text':       '#e6edf3',
}

# ── Primitives ─────────────────────────────────────────────────────────────
def box(cx, cy, w, h, fc, ec, lw=2.0, r=0.015, z=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad=0,rounding_size={r}',
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def group(cx, cy, w, h, ec, title, tsz=10, z=1):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0,rounding_size=0.022',
        facecolor=ec, edgecolor=ec, linewidth=2.5, alpha=0.14, zorder=z))
    ax.text(cx - w/2 + 0.018, cy + h/2 - 0.013, title,
            ha='left', va='top', fontsize=tsz, color=ec,
            fontweight='bold', alpha=0.90, fontfamily='monospace', zorder=z + 1)

def T(cx, cy, s, sz=11, fw='bold', c='#e6edf3', z=5):
    ax.text(cx, cy, s, ha='center', va='center',
            fontsize=sz, fontweight=fw, color=c, zorder=z)

def S(cx, cy, s, sz=7.8, c='#8b949e', z=5):
    ax.text(cx, cy, s, ha='center', va='center',
            fontsize=sz, color=c, zorder=z)

def arr(x1, y1, x2, y2, c='#8b949e', lw=1.6, conn='arc3,rad=0.0', ms=16, z=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                        connectionstyle=conn, mutation_scale=ms), zorder=z)

def lbl(cx, cy, s, c='#8b949e', sz=7.4, z=8):
    ax.text(cx, cy, s, ha='center', va='center', fontsize=sz, color=c, zorder=z,
            bbox=dict(boxstyle='round,pad=0.25', facecolor=BG, edgecolor='none', alpha=0.95))

def divider(y, c='#21262d', lw=1.0):
    ax.axhline(y, color=c, linewidth=lw, zorder=0, alpha=0.6)

# ── Layout ─────────────────────────────────────────────────────────────────
# Y-axis positions (0 = bottom, 1 = top)
YH   = 0.962  # Human
YFE  = 0.882  # Frontend row
YPE  = 0.756  # Persistence row
YORC = 0.563  # Orchestration / Tools
YHRN = 0.374  # Harness
YAGT = 0.194  # Core Agent
YLLM = 0.062  # LLM Provider

# X-axis positions
XH    = 0.500
XCNV  = 0.183;  XRUN = 0.500;  XSET = 0.817
XSCH  = 0.266;  XSTS = 0.734
XORC  = 0.217;  XTOL = 0.783
XHR   = 0.500
XAGT  = 0.500
XLLM  = 0.500

# Box [width, height]
D = {
    'h':   (0.160, 0.048),
    'fe':  (0.248, 0.076),
    'pe':  (0.328, 0.094),   # User Settings
    'sch': (0.328, 0.132),   # Workflow Schema — taller for three named subsets
    'or':  (0.308, 0.150),
    'tl':  (0.308, 0.150),
    'hr':  (0.655, 0.106),
    'ag':  (0.500, 0.094),
    'll':  (0.196, 0.048),
}

def hw(k):   return D[k][0]
def hh(k):   return D[k][1]

# ── Group backgrounds ──────────────────────────────────────────────────────
group(0.500, YFE,   0.968, 0.112, P['g_fe'], '1  FRONTEND', 10)
group(0.500, YPE,   0.968, 0.146, P['g_pe'], '2  SCHEMA & SETTINGS', 10)
group(0.500, 0.346, 0.968, 0.638, P['g_be'], '3  BACKEND', 10)

# ── Human ──────────────────────────────────────────────────────────────────
box(XH, YH, *D['h'], P['human_bg'], P['human'], lw=2.8)
T(XH, YH, 'Human', sz=14)

# ── Frontend ───────────────────────────────────────────────────────────────
# Canvas
box(XCNV, YFE, *D['fe'], P['fe_bg'], P['fe'])
T(XCNV, YFE + 0.017, 'Agent Canvas', sz=10)
S(XCNV, YFE + 0.000, 'drag-and-drop workflow design')
S(XCNV, YFE - 0.014, 'create · edit · validate schema')

# Run Viewer
box(XRUN, YFE, *D['fe'], P['fe_bg'], P['fe'])
T(XRUN, YFE + 0.017, 'Run Viewer', sz=10)
S(XRUN, YFE + 0.000, 'live streaming output per agent node')
S(XRUN, YFE - 0.014, 'run history · SSE')

# Settings UI
box(XSET, YFE, *D['fe'], P['fe_bg'], P['fe'])
T(XSET, YFE + 0.017, 'User Settings UI', sz=10)
S(XSET, YFE + 0.000, 'API credentials · LLM provider')
S(XSET, YFE - 0.014, 'source endpoints · defaults')

# ── Persistence ────────────────────────────────────────────────────────────
# Workflow Schema — 3 subsets: Topology (left), Workflow Config (right top), Node Config (right bottom)
box(XSCH, YPE, *D['sch'], P['pe_bg'], P['pe'])
T(XSCH, YPE + 0.052, 'Workflow Schema', sz=10.5, c='#aad9aa')
# Vertical divider: left (Topology) vs right (Workflow Config / Node Config)
ax.plot([XSCH, XSCH], [YPE + 0.047, YPE - 0.061],
        color=P['pe'], lw=0.6, alpha=0.45, zorder=4)
# Horizontal divider: right half only — Workflow Config vs Node Config
ax.plot([XSCH + 0.012, XSCH + hw('sch')/2 - 0.012], [YPE, YPE],
        color=P['pe'], lw=0.6, alpha=0.45, zorder=4)
# Left half: Topology
S(XSCH - hw('sch')/4, YPE + 0.035, 'Topology', sz=8.5, c='#aad9aa')
S(XSCH - hw('sch')/4, YPE + 0.019, 'nodes', sz=7.3)
S(XSCH - hw('sch')/4, YPE + 0.003, 'edges', sz=7.3)
S(XSCH - hw('sch')/4, YPE - 0.013, 'tool bindings', sz=7.3)
# Right top: Workflow Config
S(XSCH + hw('sch')/4, YPE + 0.035, 'Workflow Config', sz=8.5, c='#aad9aa')
S(XSCH + hw('sch')/4, YPE + 0.021, 'total_timeout · logging_level', sz=7.3)
S(XSCH + hw('sch')/4, YPE + 0.007, 'trace_enabled · max_loop_rounds · max_iterations', sz=7.3)
# Right bottom: Node Config
S(XSCH + hw('sch')/4, YPE - 0.010, 'Node Config', sz=8.5, c='#aad9aa')
S(XSCH + hw('sch')/4, YPE - 0.023, 'LLM params · response format', sz=7.3)
S(XSCH + hw('sch')/4, YPE - 0.035, 'agent rules · retries', sz=7.3)
S(XSCH + hw('sch')/4, YPE - 0.047, 'termination_conditions · tools', sz=7.3)
S(XSCH + hw('sch')/4, YPE - 0.059, 'parallelism · model_selection', sz=7.3)

# User Settings
box(XSTS, YPE, *D['pe'], P['pe_bg'], P['pe'])
T(XSTS, YPE + 0.026, 'User Settings', sz=10.5, c='#aad9aa')
S(XSTS, YPE + 0.003, 'API keys · LLM provider settings')
S(XSTS, YPE - 0.020, 'rate limits · named source endpoints')

# ── Backend: Orchestration ─────────────────────────────────────────────────
box(XORC, YORC, *D['or'], P['orch_bg'], P['orch'], lw=2.3)
T(XORC, YORC + 0.057, '3b  Orchestration', sz=10.5, c='#d2a8ff')
S(XORC, YORC + 0.030, 'parse schema → agent orchestration')
S(XORC, YORC + 0.008, 'validate topology · type-check edges')
S(XORC, YORC - 0.015, 'resolve logical conflicts · warn')
S(XORC, YORC - 0.039, 'assign tool bindings · determine parallelism')
S(XORC, YORC - 0.057, 'emit HOW to Harness')

# ── Backend: Agentic Tools ─────────────────────────────────────────────────
box(XTOL, YORC, *D['tl'], P['tools_bg'], P['tools'], lw=2.3)
T(XTOL, YORC + 0.057, '3a  Agentic Tools', sz=10.5, c='#f0c96c')
S(XTOL, YORC + 0.030, 'market data · social · news · filings')
S(XTOL, YORC + 0.008, 'TA indicators · alpha/beta · benchmarks')
S(XTOL, YORC - 0.015, 'large corpus chunking & retrieval')
S(XTOL, YORC - 0.039, 'semantic search · time-series utils')
S(XTOL, YORC - 0.057, 'alerts · webhooks · external write')

# ── Backend: Harness ──────────────────────────────────────────────────────
box(XHR, YHRN, *D['hr'], P['harness_bg'], P['harness'], lw=2.3)
T(XHR, YHRN + 0.034, '3d  Harness', sz=10.5, c='#ffa198')
# Divider between the two subsets
ax.plot([XHR - hw('hr')/2 + 0.025, XHR + hw('hr')/2 - 0.025], [YHRN, YHRN],
        color=P['harness'], lw=0.6, alpha=0.45, zorder=4)
S(XHR, YHRN + 0.020, 'Context Harness', sz=8.5, c='#ffa198')
S(XHR, YHRN + 0.007,
  'HOW + WHAT  →  injectable prompts + mechanical constraints  (injection into agent)')
S(XHR, YHRN - 0.010, 'Execution Harness', sz=8.5, c='#ffa198')
S(XHR, YHRN - 0.025,
  'resolve tool calls · route to 3a · enforce execution constraints · return results to agent')

# ── Backend: Core Agent ────────────────────────────────────────────────────
box(XAGT, YAGT, *D['ag'], P['agent_bg'], P['agent'], lw=2.3)
T(XAGT, YAGT + 0.020, '3c  Core Agent', sz=10.5, c='#a5d6ff')
S(XAGT, YAGT - 0.005,
  'receive assembled context  ·  resolve output schema  ·  call LLM  ·  return structured output  ·  stream via SSE')

# ── LLM Provider ──────────────────────────────────────────────────────────
box(XLLM, YLLM, *D['ll'], P['llm_bg'], P['llm'], lw=2.0)
T(XLLM, YLLM, 'LLM Provider', sz=10, fw='normal', c='#c9d1d9')

# ── ARROWS ─────────────────────────────────────────────────────────────────
FE  = P['fe'];     GR = P['pe'];    OR = P['orch']
TC  = P['tools'];  HC = P['harness']; AG = P['agent']
LC  = P['llm'];    OL = '#b08bff';  SS = '#79c0ff'

# 1 · Human ↔ Canvas
arr(XH - 0.055, YH - hh('h')/2,  XCNV + 0.060, YFE + hh('fe')/2,
    c=FE, conn='arc3,rad=-0.13', lw=1.9)
arr(XCNV + 0.060, YFE + hh('fe')/2,  XH - 0.055, YH - hh('h')/2,
    c=FE, conn='arc3,rad=0.13', lw=1.9)
lbl(0.338, 0.926, 'design / trigger / inspect', FE)

# 2 · Human ↔ Settings UI
arr(XH + 0.055, YH - hh('h')/2,  XSET - 0.060, YFE + hh('fe')/2,
    c=FE, conn='arc3,rad=0.13', lw=1.9)
arr(XSET - 0.060, YFE + hh('fe')/2,  XH + 0.055, YH - hh('h')/2,
    c=FE, conn='arc3,rad=-0.13', lw=1.9)
lbl(0.662, 0.926, 'configure', FE)

# 3 · Canvas ↔ Workflow Schema
arr(XCNV - 0.010, YFE - hh('fe')/2,  XSCH + 0.025, YPE + hh('sch')/2,
    c=GR, conn='arc3,rad=0.05', lw=1.9)
arr(XSCH + 0.025, YPE + hh('sch')/2,  XCNV - 0.010, YFE - hh('fe')/2,
    c=GR, conn='arc3,rad=-0.05', lw=1.9)
lbl(0.194, 0.822, 'read / write schema', GR)

# 4 · Settings UI ↔ Settings Store
arr(XSET + 0.010, YFE - hh('fe')/2,  XSTS - 0.025, YPE + hh('pe')/2,
    c=GR, conn='arc3,rad=-0.05', lw=1.9)
arr(XSTS - 0.025, YPE + hh('pe')/2,  XSET + 0.010, YFE - hh('fe')/2,
    c=GR, conn='arc3,rad=0.05', lw=1.9)
lbl(0.806, 0.822, 'read / write', GR)

# 5 · Schema → Orchestration
arr(XSCH - 0.010, YPE - hh('sch')/2,
    XORC + 0.018, YORC + hh('or')/2,
    c=OR, lw=2.4)
lbl(0.186, 0.657, 'workflow config\n(read at execution)', OR)

# 6 · Settings → Tools
arr(XSTS + 0.010, YPE - hh('pe')/2,
    XTOL - 0.018, YORC + hh('tl')/2,
    c=TC, lw=2.4)
lbl(0.814, 0.657, 'inject\ncredentials', TC)

# 7 · Settings Store → Core Agent  (inject LLM config, far-right curve)
arr(XSTS + hw('pe')/2,           YPE - 0.010,
    XAGT + hw('ag')/2 + 0.008,  YAGT,
    c=GR, lw=1.5, conn='arc3,rad=-0.44', ms=14)
lbl(0.988, 0.453, 'inject\nLLM\nconfig', GR, sz=7.0)

# 9 · Run Viewer ↔ Core Agent  (SSE stream)
arr(XRUN - 0.025, YFE - hh('fe')/2,
    XAGT - 0.080, YAGT + hh('ag')/2,
    c='#4a90d9', lw=1.5, conn='arc3,rad=0.22', ms=14)
arr(XAGT - 0.080, YAGT + hh('ag')/2,
    XRUN - 0.025, YFE - hh('fe')/2,
    c='#4a90d9', lw=1.5, conn='arc3,rad=-0.22', ms=14)
lbl(0.368, 0.572, 'SSE\nstream', '#4a90d9')

# 10 · Orchestration → Harness  (HOW)
arr(XORC + hw('or') * 0.25, YORC - hh('or')/2,
    XHR - hw('hr') * 0.24,  YHRN + hh('hr')/2,
    c=OR, lw=3.0, ms=18)
lbl(0.310, 0.462,
    'HOW: structure · scope\nwindow · state · guardrails', OR, sz=7.8)

# 11 · Tools → Harness  (WHAT)
arr(XTOL - hw('tl') * 0.25, YORC - hh('tl')/2,
    XHR + hw('hr') * 0.24,  YHRN + hh('hr')/2,
    c=TC, lw=3.0, ms=18)
lbl(0.690, 0.462,
    'WHAT: fetched data\ncomputed analytics', TC, sz=7.8)

# 12 · Harness → Core Agent  (assembled context)
arr(XHR + 0.025, YHRN - hh('hr')/2,
    XAGT + 0.025, YAGT + hh('ag')/2,
    c=HC, lw=3.0, ms=18)
lbl(0.588, 0.280,
    'injectable prompts\n+ mechanical constraints', HC, sz=7.8)

# 13 · Core Agent ↔ LLM Provider
arr(XAGT, YAGT - hh('ag')/2,
    XLLM, YLLM + hh('ll')/2,
    c=LC, lw=2.4)
arr(XLLM + 0.010, YLLM + hh('ll')/2,
    XAGT + 0.010, YAGT - hh('ag')/2,
    c=LC, lw=2.4, conn='arc3,rad=0.06')
lbl(0.570, 0.127, 'LLM call / response', LC)

# 14 · User Settings → Orchestration
arr(XSTS - hw('pe')/4,           YPE - hh('pe')/2,
    XORC + hw('or') * 0.42,      YORC + hh('or') * 0.32,
    c=GR, lw=1.8, conn='arc3,rad=-0.22', ms=14)
lbl(0.510, 0.644, 'user settings', GR, sz=7.2)

# 15 · Schema → Core Agent  (agent config — routed at z=1, behind Harness)
arr(XSCH + 0.020,            YPE - hh('sch')/2,
    XAGT - hw('ag') * 0.30,  YAGT + hh('ag')/2,
    c=GR, lw=1.6, ms=14, z=1)
lbl(0.475, 0.676, 'agent config', GR, sz=7.2)

# 16 · Core Agent → Harness  (tool call — upward, left offset)
arr(XAGT - 0.048, YAGT + hh('ag')/2,
    XHR  - 0.048, YHRN - hh('hr')/2,
    c=HC, lw=1.8, ms=14)
lbl(0.388, 0.281, 'tool call', HC, sz=7.0)

# 17 · Harness → Tools  (tool call — upward, right offset)
arr(XHR  + hw('hr') * 0.24 + 0.020, YHRN + hh('hr')/2,
    XTOL - hw('tl') * 0.25 + 0.020, YORC - hh('tl')/2,
    c=TC, lw=1.8, ms=14)
lbl(0.760, 0.453, 'tool call', TC, sz=7.0)

# ── Title ──────────────────────────────────────────────────────────────────
ax.text(0.500, 0.995,
        'price_monitor_agent  ·  Module Architecture',
        ha='center', va='top', fontsize=18, fontweight='bold',
        color='#c9d1d9', fontfamily='monospace', zorder=10)

# ── Data flow annotation (bottom) ──────────────────────────────────────────
ax.text(0.500, 0.020,
        'Schema is the shared contract.  '
        'Tools are runtime calls, not infrastructure.  '
        'Harness separates context engineering from execution.',
        ha='center', va='center', fontsize=8, color='#484f58',
        style='italic', zorder=10)

plt.savefig(OUT, dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"Saved → {OUT}")
