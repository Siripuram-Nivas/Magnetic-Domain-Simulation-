import streamlit as st
import numpy as np
import time
import pandas as pd
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

DEFAULTS = {
    "N": 100,
    "T": 2.27,
    "J": 1.0,
    "H": 0.0,
    "steps": 10,
    "delay": 50
}

# Colors for Video-Smooth Rendering (Red/Blue Domains)
COLOR_SPIN_UP = [255, 69, 0]    # Red-Orange (#FF4500)
COLOR_SPIN_DOWN = [30, 144, 255] # Dodger Blue (#1E90FF)

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
# Completely bypasses Matplotlib overhead to eliminate all flickering.
def render_lattice_to_image(spins):
    if HAS_CUPY:
        spins = spins.get()
        
    h, w = spins.shape
    img_rgb = np.empty((h, w, 3), dtype=np.uint8)
    img_rgb[spins == 1] = COLOR_SPIN_UP
    img_rgb[spins == -1] = COLOR_SPIN_DOWN
    return img_rgb

# ----------------- UI Sidebar -----------------
with st.sidebar:
    st.title("🧲 Domain Simulation")
    st.info(f"**Physics Engine:** {'✅ GPU (CuPy)' if HAS_CUPY else '🚀 CPU (NumPy)'}")
    
    st.header("Controls & Parameters")
    N = st.slider("Grid Size (N)", 10, 150, value=st.session_state.params["N"], step=10)
    st.session_state.params["N"] = N
    
    # Auto-rebuild grid tightly if dimensions safely change
    if N != st.session_state.current_N:
        st.session_state.spins = initialize_grid(N)
        st.session_state.step_count = 0
        st.session_state.history_E.clear()
        st.session_state.history_M.clear()
        st.session_state.history_steps.clear()
        st.session_state.current_N = N

    T = st.slider("Temperature (T)", 0.1, 5.0, value=st.session_state.params["T"], step=0.05)
    J = st.slider("Interaction (J)", 0.5, 2.0, value=st.session_state.params["J"], step=0.1)
    H = st.slider("External Field (H)", -2.0, 2.0, value=st.session_state.params["H"], step=0.1)
    
    st.session_state.params["T"] = T
    st.session_state.params["J"] = J
    st.session_state.params["H"] = H
    
    st.markdown("### Rendering Engine")
    steps_per_frame = st.slider("Monte Carlo Sweeps / Frame", 1, 100, value=st.session_state.params["steps"], step=5)
    delay_ms = st.slider("Frame Delay (ms)", 0, 500, value=st.session_state.params["delay"], step=5)
    
    st.session_state.params["steps"] = steps_per_frame
    st.session_state.params["delay"] = delay_ms

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        # Use Toggle switch instead of Button for bulletproof running loops
        running_toggle = st.toggle("▶ RUN", value=st.session_state.running)
        st.session_state.running = running_toggle

    with col2:
        if st.button("🔄 Reset Global", use_container_width=True):
            st.session_state.params = DEFAULTS.copy()
            st.session_state.spins = initialize_grid(DEFAULTS["N"])
            st.session_state.step_count = 0
            st.session_state.history_E.clear()
            st.session_state.history_M.clear()
            st.session_state.history_steps.clear()
            st.session_state.current_N = DEFAULTS["N"]
            st.session_state.running = False
            st.rerun()

    show_phase = st.toggle("📊 Phase Target Analysis")

# ----------------- Modern Main Layout -----------------
col_sim, col_metrics = st.columns([1.2, 1])

# Initial values setup
N_sq = N ** 2
current_E = calculate_energy(st.session_state.spins, J, H) / N_sq if not st.session_state.history_E else st.session_state.history_E[-1]
current_M = calculate_magnetization(st.session_state.spins) / N_sq if not st.session_state.history_M else st.session_state.history_M[-1]

with col_sim:
    st.markdown(f"**Lattice Dynamics Matrix — ({N}x{N})**")
    grid_placeholder = st.empty()
    # Fast Native Canvas Push
    grid_placeholder.image(render_lattice_to_image(st.session_state.spins), use_container_width=True, output_format="PNG")

with col_metrics:
    st.markdown("**Real-Time Telemetry**")
    m1, m2, m3 = st.columns(3)
    m1.metric("MC Sweeps", f"{st.session_state.step_count:,}")
    m2.metric("Energy E(H)", f"{current_E:.3f}")
    m3.metric("Magnetization M", f"{current_M:.3f}")
    
    chart_e = st.empty()
    chart_m = st.empty()
    
    if len(st.session_state.history_E) > 1:
        # Vectorized rendering of historical traces natively via Pandas DataFrame
        df = pd.DataFrame({
            "Steps": st.session_state.history_steps,
            "Energy": st.session_state.history_E,
            "Magnetization": st.session_state.history_M
        }).set_index("Steps")
        
        chart_e.line_chart(df["Energy"], height=200, color="#FF4500")
        chart_m.line_chart(df["Magnetization"], height=200, color="#1E90FF")
    else:
        chart_e.info("Waiting for core simulation telemetry...")

# Static Heavy Duty Analytics 
if show_phase:
    st.markdown("---")
    st.subheader("📊 Analytical Target (Magnetization against Temperature)")
    
    @st.cache_data
    def get_phase_data():
        import matplotlib.pyplot as plt
        temps, mags = compute_phase_transition(N=30, sweeps=100, samples=30)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(temps, mags, 'o-', color='crimson')
        ax.axvline(2.269, color='k', linestyle='--', label='Critical T ≈ 2.27')
        ax.set_xlabel("Temperature (T)")
        ax.set_ylabel("Magnetization (|M|/N²)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig
        
    st.pyplot(get_phase_data())

# ----------------- Non-Blocking Engine Loop -----------------
# Properly yields architecture threads via st.rerun so UI Sliders work cleanly concurrently!
if st.session_state.running:
    
    # Process Frame Mathematics
    for _ in range(steps_per_frame):
        st.session_state.spins = monte_carlo_step(st.session_state.spins, T, J, H)
    st.session_state.step_count += steps_per_frame
    
    # Process Metrics
    E = calculate_energy(st.session_state.spins, J, H) / N_sq
    M = calculate_magnetization(st.session_state.spins) / N_sq
    
    st.session_state.history_steps.append(st.session_state.step_count)
    st.session_state.history_E.append(E)
    st.session_state.history_M.append(M)
    
    # Cap History Array
    if len(st.session_state.history_E) > 100:
        st.session_state.history_steps.pop(0)
        st.session_state.history_E.pop(0)
        st.session_state.history_M.pop(0)
        
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
        
    # Flushes backend immediately and processes updates synchronously
    st.rerun()
