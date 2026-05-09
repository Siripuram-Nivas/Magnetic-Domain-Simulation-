import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

def get_xp():
    return cp if HAS_CUPY else np

def initialize_grid(N, seed=None):
    xp = get_xp()
    if seed is not None:
        if HAS_CUPY:
            cp.random.seed(seed)
        np.random.seed(seed)
    return xp.random.choice([-1, 1], size=(N, N)).astype(xp.int8)

def calculate_energy(grid, J, H):
    xp = get_xp()
    # E = -J * sum(Si * Sj) - H * sum(Si)
    # Interaction energy
    interaction = grid * (xp.roll(grid, 1, axis=0) +
                          xp.roll(grid, -1, axis=0) +
                          xp.roll(grid, 1, axis=1) +
                          xp.roll(grid, -1, axis=1))
    
    # We divide by 2 so we don't double count pairs
    energy = -J * xp.sum(interaction) / 2.0
    energy -= H * xp.sum(grid)
    return float(energy)

def calculate_magnetization(grid):
    xp = get_xp()
    return float(xp.sum(grid))

def monte_carlo_step(grid, T, J, H, beta=None):
    """
    Perform one Monte Carlo sweep using checkerboard update 
    for fast NumPy/CuPy vectorization.
    """
    xp = get_xp()
    N = grid.shape[0]
    
    # Create checkerboard masks if we haven't already
    # For a step, we update white squares then black squares
    x, y = xp.meshgrid(xp.arange(N), xp.arange(N))
    mask_even = (x + y) % 2 == 0
    mask_odd = (x + y) % 2 == 1
    
    beta_val = 1.0 / T if beta is None else beta
    
    for mask in [mask_even, mask_odd]:
        # Compute sum of nearest neighbors
        neighbors = (xp.roll(grid, 1, axis=0) +
                     xp.roll(grid, -1, axis=0) +
                     xp.roll(grid, 1, axis=1) +
                     xp.roll(grid, -1, axis=1))
        
        # dE is the energy change if we flip the spin:
        # E_old = - (J * neighbors + H) * spin
        # E_new = - (J * neighbors + H) * (-spin) = (J * neighbors + H) * spin
        # dE = E_new - E_old = 2 * spin * (J * neighbors + H)
        dE = 2 * grid * (J * neighbors + H)
        
        # Calculate transition probabilities
        # We only flip if dE <= 0 OR random < exp(-dE * beta)
        # We can compute transition prob for all cells in the mask
        boltzmann = xp.exp(-dE * beta_val)
        
        # Random matrix
        rand_vals = xp.random.rand(N, N)
        
        # Flip condition
        flip_cond = mask & ((dE <= 0) | (rand_vals < boltzmann))
        
        # Apply flips
        grid = xp.where(flip_cond, -grid, grid)
        
    return grid

def compute_phase_transition(N=50, sweeps=100, samples=50):
    """
    Compute magnetization curve for temperature range [1.0, 4.0]
    to show the critical temperature phase transition.
    """
    xp = get_xp()
    T_range = np.linspace(1.0, 4.0, samples)
    magnetizations = []
    
    for T in T_range:
        grid = initialize_grid(N)
        # thermalize
        for _ in range(sweeps):
            grid = monte_carlo_step(grid, T, 1.0, 0.0)
        
        # sample
        mag_sum = 0
        for _ in range(20):
            grid = monte_carlo_step(grid, T, 1.0, 0.0)
            mag = float(xp.abs(xp.sum(grid))) / (N * N)
            mag_sum += mag
            
        magnetizations.append(mag_sum / 20.0)
        
    return T_range, magnetizations
