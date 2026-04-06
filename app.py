"""
PQC Interactive Demo — Three-Slide Educational Experience
=========================================================
Quantum math: pure numpy (instant interactivity).
Circuit visualization: PennyLane draw_mpl (Slides 1 & 3).
Slide 1: The Magic Recipe with Knobs
Slide 2: The Training Loop — animated loss landscape
Slide 3: You Are the Optimizer — match the waveform
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PQC Interactive Demo",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding: 0.5rem 2rem 1rem; max-width: 1400px; }
  #MainMenu, header, footer { visibility: hidden; }

  /* Slide header */
  .shdr { text-align: center; margin-bottom: 0.8rem; }
  .shdr h1 { font-size: 2.0rem; font-weight: 800; color: #1565C0; margin: 0; }
  .shdr .sub { font-size: 1.0rem; color: #555; margin-top: 0.2rem; }

  /* Info cards */
  .card { border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.5rem; }
  .cb { background: #e8f0fe; border-left: 4px solid #1565C0; }
  .co { background: #fff3e0; border-left: 4px solid #FB8C00; }
  .cg { background: #e8f5e9; border-left: 4px solid #2E7D32; }
  .cr { background: #fce4ec; border-left: 4px solid #C62828; }

  /* Step badge */
  .bdg { display: inline-block; background: #1565C0; color: white;
         border-radius: 50%; width: 24px; height: 24px; text-align: center;
         line-height: 24px; font-weight: 700; font-size: .85rem; margin-right: 6px; }

  /* Match meter */
  .meter { border-radius: 12px; padding: 0.8rem 1.2rem; margin-bottom: 0.8rem; }

  /* Win banner */
  .win { background: linear-gradient(135deg, #1565C0, #2E7D32); color: white;
         border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
         font-size: 1.05rem; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  QUANTUM MATH  (numpy only — no PennyLane needed, everything is instant)
# ─────────────────────────────────────────────────────────────────────────────

def _ry(t):
    """2×2 RY rotation matrix."""
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])

# Pre-build 8×8 CNOT matrices (2-qubit CNOT tensored with I)
_CNOT4 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=float)
_I2    = np.eye(2)
_C01   = np.kron(_CNOT4, _I2)  # CNOT(q0→q1) in 3-qubit space
_C12   = np.kron(_I2, _CNOT4)  # CNOT(q1→q2) in 3-qubit space

def psi3(A, B, C):
    """
    Compute the 3-qubit output state:
        |ψ⟩ = CNOT₁₂ · CNOT₀₁ · [RY(A) ⊗ RY(B) ⊗ RY(C)] |000⟩
    Returns a complex array of length 8.
    """
    psi = np.kron(_ry(A), np.kron(_ry(B), _ry(C)))[:, 0]  # apply to |000⟩
    return _C12 @ (_C01 @ psi)

def fidelity(a, b):
    """Quantum fidelity |⟨a|b⟩|²."""
    return float(abs(np.vdot(a, b)) ** 2)

# ── Slide 2: 1D pedagogical loss landscape ───────────────────────────────────

def _loss_1d(x):
    """
    Simple 1-parameter loss with 3 local minima and 1 clear global minimum.
    Built from Gaussian wells so the gradient is smooth everywhere.
      Global min ≈ 0.15  at x ≈ 3.5
      Local  min ≈ 0.55  at x ≈ 1.6
      Local  min ≈ 0.50  at x ≈ 6.2   ← optimizer gets stuck here
      Local  min ≈ 0.60  at x ≈ 8.8
    Domain: x ∈ [0, 10]
    """
    return (
        1.0
        - 0.45 * np.exp(-0.5 * ((x - 1.6) / 0.55) ** 2)   # local
        - 0.85 * np.exp(-0.5 * ((x - 3.5) / 0.65) ** 2)   # GLOBAL
        - 0.50 * np.exp(-0.5 * ((x - 6.2) / 0.60) ** 2)   # local
        - 0.40 * np.exp(-0.5 * ((x - 8.8) / 0.50) ** 2)   # local
    )

@st.cache_data(show_spinner=False)
def _precompute_1d():
    """Curve + gradient-descent path that falls into a local minimum."""
    xs = np.linspace(0, 10, 600)
    ys = _loss_1d(xs)

    # Gradient descent from x=7.5 → converges to local min at ~6.2
    h  = 1e-5
    lr = 0.12
    x  = 7.5
    px, py = [x], [float(_loss_1d(x))]
    for _ in range(70):
        grad = (_loss_1d(x + h) - _loss_1d(x - h)) / (2 * h)
        x    = float(np.clip(x - lr * grad, 0.0, 10.0))
        px.append(x)
        py.append(float(_loss_1d(x)))

    return xs, ys, np.array(px), np.array(py)

# ── Slide 3: target state ─────────────────────────────────────────────────────

_A_TARGET = np.pi / 3          # ≈ 1.047 rad
_B_TARGET = np.pi / 2          # ≈ 1.571 rad
_C_TARGET = 2 * np.pi / 3     # ≈ 2.094 rad
_PSI_TARGET = psi3(_A_TARGET, _B_TARGET, _C_TARGET)
_BASIS = [f"|{format(i, '03b')}⟩" for i in range(8)]
_CIRC  = np.linspace(0, 2 * np.pi, 60)    # for drawing circles

# ── PennyLane circuit (visualization only — quantum math stays in numpy) ──────

@st.cache_resource(show_spinner=False)
def _build_pqc():
    """Build and cache the PennyLane QNode matching the numpy circuit."""
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.RY(params[2], wires=2)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.state()

    return circuit


def draw_pqc(A, B, C):
    """
    Return a matplotlib figure of the PQC using qml.draw_mpl.
    Falls back to None on error so callers can degrade gracefully.
    """
    try:
        circuit = _build_pqc()
        fig, _ = qml.draw_mpl(circuit, style="pennylane", decimals=2)(
            np.array([A, B, C])
        )
        fig.set_size_inches(7, 2.4)
        fig.tight_layout()
        return fig
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_circuit(t1, t2, t3):
    """Interactive circuit diagram: 3 RY gates + CNOT₀₁ + CNOT₁₂."""
    yq     = [2.0, 1.0, 0.0]
    thetas = [t1, t2, t3]
    q_col  = ["#1565C0", "#2E7D32", "#6A1B9A"]
    shapes, traces, annots = [], [], []

    x0, xG, xC1, xC2, x1 = 0.3, 1.5, 3.2, 4.5, 5.9
    gw, gh = 0.72, 0.52

    # Qubit wires
    for y in yq:
        shapes.append(dict(type="line", x0=x0, x1=x1, y0=y, y1=y,
                           line=dict(color="#666", width=2.5)))

    # Qubit labels
    for i, (y, col) in enumerate(zip(yq, q_col)):
        annots.append(dict(x=0.1, y=y, text=f"<b style='color:{col}'>q{i}</b>",
                           showarrow=False, font=dict(size=14)))

    # RY gate boxes
    for y, t, lbl in zip(yq, thetas, ["θ₁", "θ₂", "θ₃"]):
        shapes.append(dict(type="rect",
                           x0=xG - gw/2, x1=xG + gw/2,
                           y0=y - gh/2,  y1=y + gh/2,
                           fillcolor="#1565C0", opacity=0.88,
                           line=dict(color="white", width=1.5)))
        annots.append(dict(x=xG, y=y,
                           text=f"<b>RY</b><br><span style='font-size:10px'>"
                                f"{lbl} = {t:.2f}</span>",
                           showarrow=False, font=dict(size=11, color="white"),
                           align="center"))

    # Draw a CNOT gate (control dot + ⊕ target)
    def _cnot(xc, y_ctrl, y_tgt):
        # Vertical wire
        shapes.append(dict(type="line", x0=xc, x1=xc, y0=y_ctrl, y1=y_tgt,
                           line=dict(color="#E53935", width=2, dash="dot")))
        # Control dot
        traces.append(go.Scatter(x=[xc], y=[y_ctrl], mode="markers",
                                 marker=dict(size=13, color="#E53935"),
                                 showlegend=False, hoverinfo="skip"))
        # Target ⊕: circle
        r = 0.27
        traces.append(go.Scatter(
            x=xc + r * np.cos(_CIRC),
            y=y_tgt + r * np.sin(_CIRC),
            mode="lines", line=dict(color="#E53935", width=2.2),
            showlegend=False, hoverinfo="skip"))
        # Target ⊕: cross lines
        for dx, dy in [(r, 0), (-r, 0), (0, r), (0, -r)]:
            shapes.append(dict(type="line",
                               x0=xc, x1=xc + dx,
                               y0=y_tgt, y1=y_tgt + dy,
                               line=dict(color="#E53935", width=2.5)))

    _cnot(xC1, yq[0], yq[1])
    _cnot(xC2, yq[1], yq[2])

    # Column headers
    for x, lbl in [(xG, "Rotation gates"), (xC1, "CNOT"), (xC2, "CNOT")]:
        annots.append(dict(x=x, y=2.75,
                           text=f"<i style='color:#888;font-size:11px'>{lbl}</i>",
                           showarrow=False))

    fig = go.Figure(traces)
    fig.update_layout(
        shapes=shapes, annotations=annots,
        height=200, margin=dict(l=10, r=10, t=28, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(range=[-0.2, 6.3], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(range=[-0.65, 3.1], showgrid=False, zeroline=False,
                   showticklabels=False),
        showlegend=False,
    )
    return fig


def fig_probs(state, title="Probability of measuring each basis state"):
    """Probability bar chart coloured by magnitude."""
    probs  = np.abs(state) ** 2
    normed = probs / (probs.max() + 1e-9)
    colors = [f"rgba({int(21+179*n)},{int(101+80*(1-n))},{int(192)},0.92)"
              for n in normed]
    fig = go.Figure(go.Bar(
        x=_BASIS, y=probs, marker_color=colors,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=195,
        margin=dict(l=20, r=10, t=32, b=30),
        yaxis=dict(range=[0, 1.05], title="P"),
        xaxis_title="Basis state",
        template="plotly_white",
    )
    return fig


@st.cache_data(show_spinner=False)
def fig_loss_1d_animated():
    """
    Animated 1D loss landscape for slide 2.
    The red dot starts at θ=7.5 and gradient-descends into a LOCAL minimum
    at θ≈6.2, visibly missing the ⭐ global minimum at θ≈3.5.
    Animation runs fully client-side via Plotly frames.
    """
    xs, ys, path_x, path_y = _precompute_1d()

    # ── Static traces ─────────────────────────────────────────────────────────
    curve = go.Scatter(
        x=xs, y=ys, mode="lines",
        fill="tozeroy", fillcolor="rgba(21,101,192,0.10)",
        line=dict(color="#1565C0", width=2.5),
        name="Loss curve", hoverinfo="skip",
    )
    # Global minimum marker
    x_gmin = xs[np.argmin(ys)]
    y_gmin = float(np.min(ys))
    gmin = go.Scatter(
        x=[x_gmin], y=[y_gmin], mode="markers+text",
        marker=dict(size=22, color="#FFD600", symbol="star",
                    line=dict(color="#555", width=1.5)),
        text=["  ⭐ Global Min"], textposition="middle right",
        name="Global minimum", showlegend=True, hoverinfo="skip",
    )
    # Local minimum annotation where dot gets stuck
    stuck_x, stuck_y = 6.2, float(_loss_1d(6.2))
    stuck = go.Scatter(
        x=[stuck_x], y=[stuck_y + 0.06], mode="text",
        text=["⚠️ Local Min<br>(optimizer stuck)"],
        textfont=dict(size=11, color="#C62828"),
        showlegend=False, hoverinfo="skip",
    )
    # Initial animated traces
    init_trail = go.Scatter(
        x=[path_x[0]], y=[path_y[0]], mode="markers",
        marker=dict(size=7, color="#E53935", opacity=0.5),
        showlegend=False, hoverinfo="skip",
    )
    init_dot = go.Scatter(
        x=[path_x[0]], y=[path_y[0]], mode="markers",
        marker=dict(size=18, color="#E53935", symbol="circle",
                    line=dict(color="white", width=2.5)),
        name="🔴 Optimizer", showlegend=True,
        hovertemplate="Step 0<br>θ = %{x:.3f}<br>Loss = %{y:.4f}<extra></extra>",
    )

    # ── Animation frames (update only traces 3 and 4) ─────────────────────────
    frames = []
    for k in range(len(path_x)):
        frames.append(go.Frame(
            data=[
                # Trail: all visited positions so far
                go.Scatter(
                    x=path_x[:k+1], y=path_y[:k+1], mode="markers",
                    marker=dict(size=6, color="#E53935", opacity=0.35),
                    showlegend=False, hoverinfo="skip",
                ),
                # Current position dot
                go.Scatter(
                    x=[path_x[k]], y=[path_y[k]], mode="markers",
                    marker=dict(size=18, color="#E53935", symbol="circle",
                                line=dict(color="white", width=2.5)),
                    name="🔴 Optimizer",
                    hovertemplate=(f"Step {k}<br>θ = {path_x[k]:.3f}"
                                   f"<br>Loss = {path_y[k]:.4f}<extra></extra>"),
                ),
            ],
            traces=[3, 4],
            name=str(k),
        ))

    fig = go.Figure(
        data=[curve, gmin, stuck, init_trail, init_dot],
        frames=frames,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=60, r=30, t=75, b=70),
        xaxis=dict(title="θ  (Parameter — your knob setting)",
                   range=[-0.3, 10.3], showgrid=True, gridcolor="#eee"),
        yaxis=dict(title="Loss  (lower = better)",
                   range=[-0.05, 1.15], showgrid=True, gridcolor="#eee"),
        title="Loss Landscape — the optimizer rolls downhill, but where does it end up?",
        paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", y=1.13, x=0.5, xanchor="center"),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.19, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶  Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 120, "redraw": False},
                                  "transition": {"duration": 40, "easing": "linear"},
                                  "fromcurrent": True}]),
                dict(label="⏸  Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[str(k)],
                              {"frame": {"duration": 0, "redraw": False},
                               "mode": "immediate",
                               "transition": {"duration": 0}}],
                        label=str(k))
                   for k in range(len(path_x))],
            transition=dict(duration=0),
            x=0.05, y=-0.08, len=0.9,
            currentvalue=dict(prefix="Step: ", visible=True, xanchor="center"),
        )],
    )
    return fig


def fig_progress_1d(fidelity_score):
    """
    Slide 3 companion: show the student's current position on the same 1D
    loss curve from Slide 2.  Fidelity 0 → far right (high loss);
    fidelity 1 → global minimum ⭐.  Not physically exact — purely illustrative.
    """
    xs, ys, _, _ = _precompute_1d()
    x_gmin = float(xs[np.argmin(ys)])
    x_start = 9.0

    # Linear interpolation: low fidelity = far right, high = at global min
    x_dot = x_start + fidelity_score * (x_gmin - x_start)
    y_dot = float(_loss_1d(x_dot))

    dot_color = ("#C62828" if fidelity_score < 0.4 else
                 "#FB8C00" if fidelity_score < 0.75 else
                 "#2E7D32" if fidelity_score < 0.98 else
                 "#FFD600")

    fig = go.Figure()

    # Loss curve (filled)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        fill="tozeroy", fillcolor="rgba(21,101,192,0.08)",
        line=dict(color="#1565C0", width=2),
        showlegend=False, hoverinfo="skip",
    ))

    # Global minimum star
    fig.add_trace(go.Scatter(
        x=[x_gmin], y=[float(np.min(ys))],
        mode="markers+text",
        marker=dict(size=18, color="#FFD600", symbol="star",
                    line=dict(color="#555", width=1.5)),
        text=["  ⭐ Goal"], textposition="middle right",
        showlegend=False, hoverinfo="skip",
    ))

    # Student's dot
    fig.add_trace(go.Scatter(
        x=[x_dot], y=[y_dot], mode="markers+text",
        marker=dict(size=18, color=dot_color, symbol="circle",
                    line=dict(color="white", width=2.5)),
        text=[f"  You ({fidelity_score*100:.0f}%)"],
        textposition="top right",
        textfont=dict(size=11, color=dot_color),
        showlegend=False,
        hovertemplate=f"Fidelity {fidelity_score*100:.1f}%<br>Loss ≈ {y_dot:.3f}<extra></extra>",
    ))

    # Dashed arrow from dot toward global min (only when not yet at goal)
    if fidelity_score < 0.97:
        fig.add_annotation(
            x=x_gmin + 0.4, y=float(np.min(ys)),
            ax=x_dot, ay=y_dot,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=1.5, arrowcolor="#aaa",
        )

    fig.update_layout(
        height=260,
        margin=dict(l=50, r=20, t=35, b=35),
        xaxis=dict(title="θ", range=[-0.3, 10.3],
                   showgrid=True, gridcolor="#eee", zeroline=False),
        yaxis=dict(title="Loss", range=[-0.05, 1.1],
                   showgrid=True, gridcolor="#eee", zeroline=False),
        title="📍 Your position on the loss landscape (from Slide 2)",
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig


def fig_waveform(A, B, C):
    """
    Slide 3 waveform: target vs student's current quantum state.
    Shows |amplitude|² with smooth spline interpolation for a 'wave' feel.
    """
    curr = psi3(A, B, C)
    p_tgt  = np.abs(_PSI_TARGET) ** 2
    p_curr = np.abs(curr) ** 2

    fig = go.Figure()

    # Target wave (filled area)
    fig.add_trace(go.Scatter(
        x=_BASIS, y=p_tgt,
        mode="lines+markers",
        line=dict(color="#FB8C00", width=3, shape="spline", smoothing=0.9),
        marker=dict(size=10, color="#FB8C00"),
        fill="tozeroy", fillcolor="rgba(251,140,0,0.15)",
        name="🎯 Target",
        hovertemplate="%{x}: %{y:.3f}<extra>Target</extra>",
    ))

    # Student's current wave (filled area)
    fig.add_trace(go.Scatter(
        x=_BASIS, y=p_curr,
        mode="lines+markers",
        line=dict(color="#1565C0", width=3, shape="spline", smoothing=0.9),
        marker=dict(size=10, color="#1565C0"),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.15)",
        name="⚡ Yours",
        hovertemplate="%{x}: %{y:.3f}<extra>Yours</extra>",
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=30, r=20, t=45, b=35),
        yaxis=dict(range=[0, 1.05], title="Probability"),
        xaxis_title="Basis State",
        legend=dict(orientation="h", y=1.16, x=0.5, xanchor="center"),
        template="plotly_white",
        title="Match Your Wave to the Target!  (Orange = Target, Blue = Yours)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE BOOT
# ─────────────────────────────────────────────────────────────────────────────
if "slide" not in st.session_state:
    st.session_state["slide"] = 0

# ─────────────────────────────────────────────────────────────────────────────
#  NAVIGATION BAR  (top + bottom)
# ─────────────────────────────────────────────────────────────────────────────
_TITLES = [
    "🍪  The Magic Recipe",
    "🤖  Watch the Robot Learn",
    "🎮  You Are the Optimizer",
]


def _nav(key_suffix="top"):
    sl = st.session_state["slide"]
    c_prev, c_mid, c_next = st.columns([1, 6, 1])

    with c_prev:
        if sl > 0:
            if st.button("← Back", key=f"prev_{key_suffix}", use_container_width=True):
                st.session_state["slide"] -= 1
                st.rerun()

    with c_mid:
        dots = ""
        for i, title in enumerate(_TITLES):
            col = "#1565C0" if i == sl else "#bbb"
            wt  = "800"    if i == sl else "400"
            dots += (f'<span style="color:{col};font-size:1.2rem;font-weight:{wt};">'
                     f'● {title}</span>')
            if i < 2:
                dots += '<span style="color:#ddd;"> &nbsp;→&nbsp; </span>'
        st.markdown(f'<div style="text-align:center;padding:0.15rem 0;">{dots}</div>',
                    unsafe_allow_html=True)

    with c_next:
        if sl < 2:
            if st.button("Next →", key=f"next_{key_suffix}",
                         use_container_width=True, type="primary"):
                st.session_state["slide"] += 1
                st.rerun()
        elif key_suffix == "top":
            st.markdown('<div style="text-align:right;color:#2E7D32;padding-top:0.4rem;">'
                        '🎓 Complete!</div>', unsafe_allow_html=True)

    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  SLIDE 1 — The Magic Recipe with Knobs
# ─────────────────────────────────────────────────────────────────────────────

def slide1():
    st.markdown("""
    <div class="shdr">
      <h1>⚛️ The Magic Recipe: Parametrized Quantum Circuits</h1>
      <p class="sub">Turn the knobs below — watch the quantum circuit and output state change live!</p>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([0.43, 0.57], gap="large")

    with left:
        st.markdown("""
        <div class="card cb">
          <b>🤖 The Robot Chef Analogy</b><br>
          Imagine a robot chef trying to bake the <em>perfect cookie</em>.
          It doesn't know the exact temperature or baking time — but it has
          <b>knobs to turn</b>. It tries a setting, tastes the result,
          and adjusts. A PQC works exactly the same way!
        </div>

        <div class="card co">
          <b>🔮 What is a Quantum Circuit?</b><br>
          A set of step-by-step instructions for a quantum computer.
          It acts on <b>qubits</b> — special particles that can be 0 and 1
          <em>at the same time</em> (superposition).
          <br>• <b>RY gate</b> — rotates a qubit by angle θ
          <br>• <b>CNOT gate</b> — entangles two qubits together
        </div>

        <div class="card co">
          <b>🎛️ What Makes it <em>Parametrized</em>?</b><br>
          The three RY gates have adjustable angles <b>θ₁, θ₂, θ₃</b>.
          Turn a knob → that qubit rotates by a different amount →
          the entire output state shifts.  These are the "recipe settings."
        </div>

        <div class="card cg">
          <b>🤝 Why Use a PQC?</b><br>
          Today's quantum computers are small and noisy.
          PQCs are a <b>teamwork solution</b>:<br>
          &nbsp;&nbsp;🔵 <em>Quantum computer</em> — runs the circuit<br>
          &nbsp;&nbsp;💻 <em>Regular computer</em> — turns the knobs to improve the answer<br>
          This partnership is called a <b>Variational Quantum Algorithm (VQA)</b>.
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("**🎛️ Turn the Knobs — Circuit and Output Update in Real Time!**")

        kc1, kc2, kc3 = st.columns(3)
        t1 = kc1.slider("θ₁  (Knob 1)", 0.0, float(2 * np.pi),
                         float(np.pi / 4), step=0.05, format="%.2f", key="s1_t1")
        t2 = kc2.slider("θ₂  (Knob 2)", 0.0, float(2 * np.pi),
                         float(np.pi / 3), step=0.05, format="%.2f", key="s1_t2")
        t3 = kc3.slider("θ₃  (Knob 3)", 0.0, float(2 * np.pi),
                         float(np.pi / 2), step=0.05, format="%.2f", key="s1_t3")

        pqc_fig = draw_pqc(t1, t2, t3)
        if pqc_fig is not None:
            st.pyplot(pqc_fig, use_container_width=True)
            plt.close(pqc_fig)
        else:
            st.plotly_chart(fig_circuit(t1, t2, t3), use_container_width=True)
        st.plotly_chart(fig_probs(psi3(t1, t2, t3)), use_container_width=True)

        st.caption(
            "💡 Tip: Try θ₁ ≈ 1.57 (π/2) and θ₂ ≈ 0 to see a Bell-like entangled state "
            "where |000⟩ and |010⟩ dominate!"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SLIDE 2 — The Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def slide2():
    st.markdown("""
    <div class="shdr">
      <h1>🤖 Watch the Robot Learn!  (The Training Loop)</h1>
      <p class="sub">Press ▶ Play on the chart to watch the optimizer roll down into the valley.</p>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([0.36, 0.64], gap="large")

    with left:
        st.markdown("""
        <div class="card cb">
          <b>🔑 Key Idea: Optimization</b><br>
          "Optimization" just means <em>finding the best knob settings</em>.
          Picture a hilly landscape — the goal is to roll downhill
          into the <b>deepest valley</b> (lowest loss).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**The loop repeats until the valley is found:**")
        st.markdown("""
        <div class="card cb" style="padding:0.5rem 0.8rem;">
          <span class="bdg">1</span><b>Guess</b> — pick a random value for θ
        </div>
        <div class="card co" style="padding:0.5rem 0.8rem;">
          <span class="bdg" style="background:#FB8C00;">2</span>
          <b>Measure</b> — run the quantum circuit, get a Loss score
        </div>
        <div class="card cg" style="padding:0.5rem 0.8rem;">
          <span class="bdg" style="background:#2E7D32;">3</span>
          <b>Adjust</b> — nudge θ downhill (one gradient step)
        </div>
        <div class="card cb" style="padding:0.5rem 0.8rem;">
          <span class="bdg" style="background:#6A1B9A;">4</span>
          <b>Repeat</b> — keep going until the dot stops moving ✅
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card cb" style="margin-top:0.6rem;">
          <b>🗺️ Reading the Chart</b><br>
          The x-axis is your single knob setting θ.<br>
          The y-axis is the Loss — <b>lower is better</b>.<br>
          The curve has <b>hills</b> (bad) and <b>valleys</b> (good).<br>
          ⭐ marks the <em>deepest</em> valley — the global minimum.
        </div>

        <div class="card cr" style="margin-top:0.4rem;">
          <b>⚠️ Watch what happens!</b><br>
          The 🔴 dot starts on the right and rolls downhill —
          but it falls into a <em>shallow valley</em> first and
          <b>gets stuck</b>, never reaching ⭐.<br>
          This is the <b>local minimum problem</b>: a core challenge
          in training PQCs!
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.plotly_chart(fig_loss_1d_animated(), use_container_width=True)
        st.caption(
            "The 🔴 dot is the optimizer following gradient descent — "
            "at each step it moves in the direction that decreases Loss. "
            "Notice it settles in the ⚠️ local minimum at θ≈6.2 "
            "instead of the ⭐ global minimum at θ≈3.5. "
            "Use the step slider to scrub through the path manually."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SLIDE 3 — You Are the Optimizer!
# ─────────────────────────────────────────────────────────────────────────────

def slide3():
    # Read values from session state BEFORE widgets render, so the meter
    # at the top already reflects the latest slider state on every rerun.
    A = st.session_state.get("s3_A", 0.0)
    B = st.session_state.get("s3_B", 0.0)
    C = st.session_state.get("s3_C", 0.0)
    fid = fidelity(psi3(A, B, C), _PSI_TARGET)
    pct = fid * 100.0

    st.markdown("""
    <div class="shdr">
      <h1>🎮 You Are the Optimizer!</h1>
      <p class="sub">Tune three angles to make <b>your wave</b> match the <b>target wave</b>.
      Fill the Match Meter to 100%!</p>
    </div>""", unsafe_allow_html=True)

    # ── Match Meter (full-width, always at top) ───────────────────────────────
    mc = ("#C62828" if pct < 40 else "#FB8C00" if pct < 75
          else "#2E7D32" if pct < 98 else "#1565C0")
    msg = ("🔴 Keep trying — you're far off!"      if pct < 40  else
           "🟠 Getting warmer — keep adjusting."   if pct < 75  else
           "🟢 So close!  Fine-tune carefully."    if pct < 98  else
           "⭐ PERFECT MATCH — you did it!")

    st.markdown(f"""
    <div class="meter" style="background:#f5f5f5;border:2px solid {mc};">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:.3rem;">
        <span style="font-weight:700;font-size:1.1rem;">🎯 Match Meter</span>
        <span style="font-size:1.5rem;font-weight:800;color:{mc};">{pct:.1f}%</span>
      </div>
      <div style="background:#ddd;border-radius:8px;height:26px;overflow:hidden;">
        <div style="background:{mc};width:{min(pct,100):.2f}%;height:100%;
                    border-radius:8px;"></div>
      </div>
      <div style="font-size:.88rem;color:#555;margin-top:.3rem;">{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Success banner ────────────────────────────────────────────────────────
    if pct >= 98:
        st.markdown("""
        <div class="win">
          🎉 <b>SUCCESS!  You just performed a Variational Quantum Algorithm!</b> 🎉<br><br>
          By adjusting three rotation angles, you acted as the <b>classical optimizer</b>
          driving a quantum circuit to match a target quantum state.<br>
          This is exactly how real algorithms like <b>VQE</b> and <b>QAOA</b> work —
          except the computer turns the knobs millions of times automatically!
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    left, right = st.columns([0.33, 0.67], gap="large")

    with left:
        st.markdown("""
        <div class="card cb">
          <b>🎛️ Your Controls</b><br>
          Adjust Angle A, B, and C so the <b>blue wave</b>
          matches the <b>orange target wave</b>.
          The Match Meter fills as you get closer!
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Angle A** — qubit 0 rotation")
        A = st.slider("A", 0.0, float(2 * np.pi), 0.0,
                      step=0.02, format="%.2f",
                      key="s3_A", label_visibility="collapsed")

        st.markdown("**Angle B** — qubit 1 rotation")
        B = st.slider("B", 0.0, float(2 * np.pi), 0.0,
                      step=0.02, format="%.2f",
                      key="s3_B", label_visibility="collapsed")

        st.markdown("**Angle C** — qubit 2 rotation")
        C = st.slider("C", 0.0, float(2 * np.pi), 0.0,
                      step=0.02, format="%.2f",
                      key="s3_C", label_visibility="collapsed")

        # Refresh fidelity after slider widgets (picks up current-run values)
        fid = fidelity(psi3(A, B, C), _PSI_TARGET)
        pct = fid * 100.0

        st.markdown("---")
        st.markdown("""
        <div class="card co">
          <b>💡 Hints</b><br>
          • Start with <b>Angle A</b> — try values around 1.0<br>
          • Then tune <b>Angle B</b> independently (try ~1.5)<br>
          • Finally fine-tune <b>Angle C</b> (try ~2.0)<br>
          • Near the solution, use tiny steps!
        </div>
        """, unsafe_allow_html=True)

        col_rnd, col_rst = st.columns(2)
        with col_rnd:
            if st.button("🎲 Random Guess", use_container_width=True):
                for k in ["s3_A", "s3_B", "s3_C"]:
                    st.session_state[k] = float(np.random.uniform(0, 2 * np.pi))
                st.rerun()
        with col_rst:
            if st.button("🔄 Reset to 0", use_container_width=True):
                for k in ["s3_A", "s3_B", "s3_C"]:
                    st.session_state[k] = 0.0
                st.rerun()

    with right:
        w_col, p_col = st.columns(2, gap="small")
        with w_col:
            st.plotly_chart(fig_waveform(A, B, C), use_container_width=True)
        with p_col:
            st.plotly_chart(fig_progress_1d(fid), use_container_width=True)

        st.markdown("**Your Circuit** *(parameterized by Angle A, B, C)*")
        pqc_fig3 = draw_pqc(A, B, C)
        if pqc_fig3 is not None:
            st.pyplot(pqc_fig3, use_container_width=True)
            plt.close(pqc_fig3)

        with st.expander("🔬 Advanced: complex amplitudes & target angles"):
            curr = psi3(A, B, C)
            fig_adv = go.Figure()
            fig_adv.add_trace(go.Bar(
                x=_BASIS, y=_PSI_TARGET.real,
                name="Target  Re(ψ)", marker_color="#FB8C00", opacity=0.70))
            fig_adv.add_trace(go.Bar(
                x=_BASIS, y=curr.real,
                name="Yours  Re(ψ)", marker_color="#1565C0", opacity=0.85))
            fig_adv.update_layout(
                barmode="overlay", height=195,
                margin=dict(l=20, r=10, t=15, b=30),
                legend=dict(orientation="h", y=1.1),
                template="plotly_white",
            )
            st.plotly_chart(fig_adv, use_container_width=True)
            st.markdown(
                f"**Target:** "
                f"A = {_A_TARGET:.3f} rad (π/3),  "
                f"B = {_B_TARGET:.3f} rad (π/2),  "
                f"C = {_C_TARGET:.3f} rad (2π/3)"
            )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

_nav("top")

{0: slide1, 1: slide2, 2: slide3}[st.session_state["slide"]]()

# Bottom navigation mirrors the top
st.divider()
_nav("bot")
