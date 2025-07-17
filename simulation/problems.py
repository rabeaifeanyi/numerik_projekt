import numpy as np

def make_masks(Nx, Ny, problem="lid_standard"):
    # mega hässlich so haha
    if problem == "lid_standard": 
        boundary = np.zeros((Ny, Nx), dtype=int)
        boundary[0,:] = boundary[-1,:] = boundary[:,0] = boundary[:,-1] = 1

        top = np.zeros((Ny, Nx), dtype=int)
        top[0, :] = 1

        walls = np.zeros((Ny, Nx), dtype=int)
        walls[-1,:] = 1
        walls[:,0] = 1
        walls[:,-1] = 1

        return {'boundary': boundary, 'top': top, 'walls': walls}
    
    else:
        raise NotImplementedError(f"Problem '{problem}' is not implemented yet :(")
