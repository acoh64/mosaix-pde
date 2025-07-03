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