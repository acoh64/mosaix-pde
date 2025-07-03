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