import matplotlib
matplotlib.use("Agg")  # Must be set BEFORE importing pyplot — prevents rendering crashes on cloud

import streamlit as st
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

from ising_simulation import (
    initialize_grid,
    monte_carlo_step,
    calculate_energy,
    calculate_magnetization,
    compute_phase_transition,
    HAS_CUPY
)

# ----------------- Professional Configuration -----------------
st.set_page_config(page_title="Ising Model Simulator", layout="wide", page_icon="🧲")

# Safe defaults — N=50 is cloud-friendly (N=100 can be slow on CPU-only)
DEFAULTS = {
    "N": 50,
    "T": 2.27,
    "J": 1.0,
    "H": 0.0,
    "steps": 5,
    "delay": 100
}

# Colors for spin rendering (Red-Orange = up, Dodger Blue = down)
COLOR_SPIN_UP   = [255, 69, 0]    # #FF4500
COLOR_SPIN_DOWN = [30, 144, 255]  # #1E90FF

# ----------------- Session State Initialisation -----------------
def init_session_state():
    if "params" not in st.session_state:
        st.session_state.params = DEFAULTS.copy()
    if "spins" not in st.session_state:
        st.session_state.spins = initialize_grid(st.session_state.params["N"])
    if "step_count" not in st.session_state:
        st.session_state.step_count = 0
    if "running" not in st.session_state:
        st.session_state.running = False
    if "history_E" not in st.session_state:
        st.session_state.history_E = []
    if "history_M" not in st.session_state:
        st.session_state.history_M = []
    if "history_steps" not in st.session_state:
        st.session_state.history_steps = []
    if "current_N" not in st.session_state:
        st.session_state.current_N = st.session_state.params["N"]

init_session_state()

# ----------------- Fast Native Renderer -----------------
# Bypasses Matplotlib entirely to eliminate flickering on every frame.
def render_lattice_to_image(spins):
    """Convert spin grid to RGB image array without Matplotlib."""
    try:
        # If CuPy array, pull back to CPU first
        if HAS_CUPY:
            import cupy as cp
            if isinstance(spins, cp.ndarray):
                spins = spins.get()
    except Exception:
        pass

    h, w = spins.shape
    img_rgb = np.empty((h, w, 3), dtype=np.uint8)
    img_rgb[spins == 1]  = COLOR_SPIN_UP
    img_rgb[spins == -1] = COLOR_SPIN_DOWN
    return img_rgb

# ----------------- UI Sidebar -----------------
with st.sidebar:
    st.title("🧲 Domain Simulation")
    engine_label = "✅ GPU (CuPy)" if HAS_CUPY else "🚀 CPU (NumPy)"
    st.info(f"**Physics Engine:** {engine_label}")

    st.header("Controls & Parameters")

    N = st.slider(
        "Grid Size (N)", 10, 150,
        value=int(st.session_state.params["N"]), step=10
    )
    st.session_state.params["N"] = N

    # Rebuild grid only when N actually changes
    if N != st.session_state.current_N:
        st.session_state.spins = initialize_grid(N)
        st.session_state.step_count = 0
        st.session_state.history_E.clear()
        st.session_state.history_M.clear()
        st.session_state.history_steps.clear()
        st.session_state.current_N = N

    T = st.slider(
        "Temperature (T)", 0.1, 5.0,
        value=float(st.session_state.params["T"]), step=0.05
    )
    J = st.slider(
        "Interaction (J)", 0.5, 2.0,
        value=float(st.session_state.params["J"]), step=0.1
    )
    H = st.slider(
        "External Field (H)", -2.0, 2.0,
        value=float(st.session_state.params["H"]), step=0.1
    )

    st.session_state.params["T"] = T
    st.session_state.params["J"] = J
    st.session_state.params["H"] = H

    st.markdown("### Rendering Engine")
    steps_per_frame = st.slider(
        "Monte Carlo Sweeps / Frame", 1, 50,
        value=int(st.session_state.params["steps"]), step=1
    )
    delay_ms = st.slider(
        "Frame Delay (ms)", 0, 500,
        value=int(st.session_state.params["delay"]), step=25
    )

    st.session_state.params["steps"] = steps_per_frame
    st.session_state.params["delay"] = delay_ms

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        running_toggle = st.toggle("▶ RUN", value=st.session_state.running)
        st.session_state.running = running_toggle

    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.params    = DEFAULTS.copy()
            st.session_state.spins     = initialize_grid(DEFAULTS["N"])
            st.session_state.step_count = 0
            st.session_state.history_E.clear()
            st.session_state.history_M.clear()
            st.session_state.history_steps.clear()
            st.session_state.current_N = DEFAULTS["N"]
            st.session_state.running   = False
            st.rerun()

    show_phase = st.toggle("📊 Phase Transition Analysis")

# ----------------- Main Layout -----------------
col_sim, col_metrics = st.columns([1.2, 1])

N_sq     = max(N ** 2, 1)  # guard against division by zero
current_E = (
    st.session_state.history_E[-1]
    if st.session_state.history_E
    else calculate_energy(st.session_state.spins, J, H) / N_sq
)
current_M = (
    st.session_state.history_M[-1]
    if st.session_state.history_M
    else calculate_magnetization(st.session_state.spins) / N_sq
)

with col_sim:
    st.markdown(f"**Lattice Dynamics Matrix — ({N}×{N})**")
    grid_placeholder = st.empty()
    grid_placeholder.image(
        render_lattice_to_image(st.session_state.spins),
        use_container_width=True,
        output_format="PNG"
    )

with col_metrics:
    st.markdown("**Real-Time Telemetry**")
    m1, m2, m3 = st.columns(3)
    m1.metric("MC Sweeps",       f"{st.session_state.step_count:,}")
    m2.metric("Energy E/N²",     f"{current_E:.4f}")
    m3.metric("Magnetization M", f"{current_M:.4f}")

    chart_e = st.empty()
    chart_m = st.empty()

    if len(st.session_state.history_E) > 1:
        df = pd.DataFrame({
            "Steps":         st.session_state.history_steps,
            "Energy":        st.session_state.history_E,
            "Magnetization": st.session_state.history_M
        }).set_index("Steps")

        chart_e.line_chart(df["Energy"],        height=200, color="#FF4500")
        chart_m.line_chart(df["Magnetization"], height=200, color="#1E90FF")
    else:
        chart_e.info("Waiting for simulation telemetry…")

# ----------------- Phase Transition Analysis (cached, Matplotlib safe) -----------------
if show_phase:
    st.markdown("---")
    st.subheader("📊 Magnetization vs. Temperature (Phase Transition)")

    @st.cache_data
    def get_phase_data():
        """Compute and render phase curve once; result is cached."""
        temps, mags = compute_phase_transition(N=30, sweeps=80, samples=25)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(temps, mags, "o-", color="crimson", linewidth=1.5, markersize=4)
        ax.axvline(2.269, color="black", linestyle="--", label="Critical T ≈ 2.27")
        ax.set_xlabel("Temperature (T)")
        ax.set_ylabel("Magnetization |M|/N²")
        ax.set_title("2D Ising Model — Phase Transition")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    phase_fig = get_phase_data()
    st.pyplot(phase_fig)
    plt.close(phase_fig)  # ✅ Prevent memory leak

# ----------------- Simulation Loop (session-state driven, no while True) -----------------
# Uses st.rerun() for frame-by-frame animation.
# Controlled by the toggle — stops automatically when toggle is off.
if st.session_state.running:

    # --- Run simulation steps ---
    try:
        for _ in range(steps_per_frame):
            st.session_state.spins = monte_carlo_step(
                st.session_state.spins, T, J, H
            )
        st.session_state.step_count += steps_per_frame
    except Exception as e:
        st.error(f"Simulation error: {e}")
        st.session_state.running = False
        st.stop()

    # --- Update metrics ---
    E = calculate_energy(st.session_state.spins, J, H) / N_sq
    M = calculate_magnetization(st.session_state.spins) / N_sq

    st.session_state.history_steps.append(st.session_state.step_count)
    st.session_state.history_E.append(E)
    st.session_state.history_M.append(M)

    # Cap history to last 100 points to prevent unbounded memory growth
    MAX_HISTORY = 100
    if len(st.session_state.history_E) > MAX_HISTORY:
        st.session_state.history_steps = st.session_state.history_steps[-MAX_HISTORY:]
        st.session_state.history_E     = st.session_state.history_E[-MAX_HISTORY:]
        st.session_state.history_M     = st.session_state.history_M[-MAX_HISTORY:]

    # Respect frame delay (capped at 500 ms for cloud safety)
    safe_delay = min(delay_ms, 500)
    if safe_delay > 0:
        time.sleep(safe_delay / 1000.0)

    # Trigger next frame — this is NOT an infinite loop;
    # Streamlit will re-enter from the top and check the toggle state.
    st.rerun()
