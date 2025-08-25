import jax.numpy as jnp
import numpy as np

# borrowed from 
# https://github.com/biswaroopmukherjee/condensate/tree/master
def initialize_Psi(N, width=100, vortexnumber=0):
    """Function for creating an initial condition for GPE
    
    Args:
        N: number of grid points
        width: width of blob
        vortextnumber: number of vortices to intialize
        
    Returns:
        initial condition for GPE simulation
    """
    psi = np.zeros((N,N), dtype=complex)
    for i in range(N):
        for j in range(N):
            phase = 1
            if vortexnumber:
                phi = vortexnumber * np.arctan2((i-N//2), (j-N//2))
                phase = np.exp(1.j * np.mod(phi,2*np.pi))
            psi[i,j] = np.exp(-( (i-N//2)/width)** 2.  -  ((j-N//2)/width)** 2. )
            psi[i,j] *= phase     
    psi = psi.astype(complex)
    return jnp.array(psi)

def add_vortex_to_wavefunction(psi, vortex_pos, vortex_strength=1, vortex_width=1):
    """Add a vortex to an existing wavefunction at a specified position
    
    Args:
        psi: existing wavefunction (complex array)
        vortex_pos: (x, y) position of the vortex center
        vortex_strength: winding number of the vortex (default=1)
        vortex_width: width of the vortex core (default=10)
        
    Returns:
        modified wavefunction with vortex added
    """
    N = psi.shape[0]
    x, y = jnp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij')
    
    # Distance from vortex center
    r = jnp.sqrt((x - vortex_pos[0])**2 + (y - vortex_pos[1])**2)
    
    # Phase (vortex winding)
    phi = vortex_strength * jnp.arctan2(y - vortex_pos[1], x - vortex_pos[0])
    
    # Create vortex phase factor
    vortex_phase = jnp.exp(1j * phi)
    
    # Create smooth transition near vortex core to avoid singularity
    # Use tanh to smoothly transition from 0 to 1 as we move away from core
    core_factor = jnp.tanh(r / vortex_width)
    
    # Combine original wavefunction with vortex
    # Near the core, use mostly vortex phase; far from core, use mostly original
    psi_with_vortex = psi * (1 - core_factor) + psi * vortex_phase * core_factor
    
    return psi_with_vortex

def detect_vortices(psi, threshold=0.1):
    """Detect the number of vortices in a wavefunction by analyzing phase winding
    
    Args:
        psi: complex wavefunction array (2D)
        threshold: minimum phase difference to consider as a vortex (default=0.1)
        
    Returns:
        tuple: (num_vortices, vortex_positions, vortex_strengths)
            - num_vortices: total number of vortices detected
            - vortex_positions: list of (x, y) positions of vortex centers
            - vortex_strengths: list of winding numbers for each vortex
    """
    if psi.ndim != 2:
        raise ValueError("Wavefunction must be a 2D array")
    
    N = psi.shape[0]
    if N != psi.shape[1]:
        raise ValueError("Wavefunction must be square")
    
    # Extract phase from complex wavefunction
    phase = jnp.angle(psi)
    
    # Initialize arrays to store vortex information
    vortex_positions = []
    vortex_strengths = []
    
    # Check each point (excluding boundaries) for vortex signature
    for i in range(1, N-1):
        for j in range(1, N-1):
            # Calculate phase differences around a small loop (2x2 pixels)
            # This gives us the phase winding around the point (i,j)
            
            # Phase differences in the four directions
            dphi_dx_plus = jnp.angle(psi[i+1, j] * jnp.conj(psi[i, j]))
            dphi_dx_minus = jnp.angle(psi[i, j] * jnp.conj(psi[i-1, j]))
            dphi_dy_plus = jnp.angle(psi[i, j+1] * jnp.conj(psi[i, j]))
            dphi_dy_minus = jnp.angle(psi[i, j] * jnp.conj(psi[i, j-1]))
            
            # Sum of phase differences around the loop
            # This should be 2π * winding_number for a vortex
            total_phase_diff = dphi_dx_plus + dphi_dy_plus - dphi_dx_minus - dphi_dy_minus
            
            # Normalize to get winding number
            winding_number = total_phase_diff / (2 * jnp.pi)
            
            # Check if this point has a significant winding number
            if abs(winding_number) > threshold:
                # This is a vortex center
                vortex_positions.append((i, j))
                vortex_strengths.append(winding_number)
    
    # Remove duplicate vortices that are very close to each other
    # (within 3 pixels of each other)
    unique_vortices = []
    unique_strengths = []
    
    for i, (pos1, strength1) in enumerate(zip(vortex_positions, vortex_strengths)):
        is_duplicate = False
        for pos2, strength2 in zip(unique_vortices, unique_strengths):
            distance = jnp.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            if distance < 3:  # 3 pixel threshold for duplicate detection
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_vortices.append(pos1)
            unique_strengths.append(strength1)
    
    num_vortices = len(unique_vortices)
    
    return num_vortices, unique_vortices, unique_strengths

def count_vortices_simple(psi):
    """Simplified vortex counting using phase gradient analysis
    
    Args:
        psi: complex wavefunction array (2D)
        
    Returns:
        int: estimated number of vortices
    """
    if psi.ndim != 2:
        raise ValueError("Wavefunction must be a 2D array")
    
    # Extract phase
    phase = jnp.angle(psi)
    
    # Calculate phase gradients
    grad_x = jnp.gradient(phase, axis=0)
    grad_y = jnp.gradient(phase, axis=1)
    
    # Calculate curl of the phase gradient (vorticity)
    # This gives us the local winding density
    curl = jnp.gradient(grad_y, axis=0) - jnp.gradient(grad_x, axis=1)
    
    # Count points with significant vorticity
    # A vortex will have high local vorticity
    threshold = 0.5 * jnp.std(curl)
    vortex_points = jnp.sum(jnp.abs(curl) > threshold)
    
    # Estimate number of vortices based on area of high vorticity
    # Each vortex typically covers several grid points
    estimated_vortices = max(1, int(vortex_points / 10))  # Rough estimate
    
    return estimated_vortices