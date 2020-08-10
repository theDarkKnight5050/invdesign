import numpy as np
from skimage.draw import circle
import constants as consts
from consts import Slice

def ppc_design(epsrs, Nx=120, Ny=120, Npml=20, space=10, num_dots=15, circle_rad=6, ra=0.6):
    # Selector for each plasma rod
    rods = [np.zeros((Nx, Ny)) for i in range(num_dots**2)]
    # Selector for entire design region
    design_region = np.zeros((Nx, Ny))
    design_region[Npml+space:Nx-Npml-space, Npml+space:Ny-Npml-space] = 1
    
    # Initialize selector for circular plasma rods
    center_x = Npml+space+circle_rad
    for i in range(num_dots):
        center_y = Npml+space+circle_rad
        for j in range(num_dots):
            r, c = circle(center_x, center_y, circle_rad*ra)
            rods[j + i*num_dots][r, c] = 1
            center_y = center_y + 2*circle_rad
        center_x = center_x + 2*circle_rad
        
    return rods, design_region
    
def init_guides(Nx=120, Ny=120, Npml=20, space=10, wg_width=12, space_slice=8, wg_shift=9):
    """Initializes waveguides and sources

    space       : The space between the PML and the structure
    wg_width    : The feed and probe waveguide width
    space_slice : The added space for the probe and source slices
    """
    bg_rho = np.zeros((Nx, Ny))
        
    # Input waveguide
    bg_rho[Nx//2-wg_width//2:Ny//2+wg_width//2,0:int(Npml+space)] = 1
    # Data waveguide 1
    bg_rho[0:int(Npml+space),Npml+space+wg_shift:Npml+space+wg_width+wg_shift] = 1
    # Data waveguide 2
    bg_rho[0:int(Npml+space),Ny-Npml-space-wg_width-wg_shift:Ny-Npml-space-wg_shift] = 1
    
    # Input probe slice
    input_slice = Slice(x=np.arange(Nx//2-wg_width//2-space_slice, Nx//2+wg_width//2+space_slice),
                        y=np.array(Npml+1))
    # Data probe slice 1
    data_slice1 = Slice(x=np.array(Npml+1),
                        y=np.arange(Npml+space-space_slice+wg_shift, Npml+space+wg_width+space_slice+wg_shift))
    # Data probe slice 2
    data_slice2 = Slice(x=np.array(Npml+1),
                        y=np.arange(Ny-Npml-space-wg_width-wg_shift-space_slice, Ny-Npml-space-wg_shift+space_slice))
    input_slices = [input_slice, data_slice1, data_slice2]
    
    # Output waveguide
    bg_rho[int(Nx-Npml-space)::,Ny//2-wg_width//2:Ny//2+wg_width//2] = 1
    # Ground waveguide
    bg_rho[Nx//2-wg_width//2:Nx//2+wg_width//2,int(Ny-Npml-space)::] = 1
    
    # Output probe slice
    output_slice = Slice(x=np.array(Nx-Npml-1),
                         y=np.arange(Ny//2-wg_width//2-space_slice, Ny//2+wg_width//2+space_slice))
    #Ground probe slice
    ground_slice = Slice(x=np.arange(Ny//2-wg_width//2-space_slice, Ny//2+wg_width//2+space_slice),
                         y=np.array(Ny-Npml-1))
    output_slices = [output_slice, ground_slice]
            
    return bg_rho, input_slices, output_slices

def init_domain(epsrs):
    rods, design_region = ppc_design(epsrs)
    bg_rho, input_slices, output_slices = \
        init_guides(consts.Nx, consts.Ny, consts.Npml, space=consts.space, wg_width=consts.wg_width, space_slice=consts.space_slice, wg_shift=9)
    return bg_rho, design_region, rods, input_slices, output_slices