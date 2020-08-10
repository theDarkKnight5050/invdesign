import numpy as np
import autograd.numpy as npa
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt
from autograd.scipy.signal import convolve as conv
from skimage.draw import circle
import ceviche
from ceviche import fdfd_ez, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode
import collections
from make_gif import make_gif


#----------------- Simulation Constants -----------------#
# Create a container for our slice coords to be used for sources and probes
Slice = collections.namedtuple('Slice', 'x y')
# The angular frequencies
omega1=2*np.pi*200e12
# Spatial resolution in meters
dl=40e-9
# Number of pixels in x-direction
Nx=120
# Number of pixels in y-direction
Ny=120
# Number of pixels in the PMLs in each direction
Npml=20
# Minimum value of the relative permittivity
epsr_min=0.0
# Maximum value of the relative permittivity
epsr_max=1.0

#-------------- Parameterization Constants --------------#
# Radius of the smoothening features
blur_radius=2
# Number of times to apply the blur
N_blur=1
# Strength of the binarizing projection
beta=10.0
# Middle point of the binarizing projection
eta=0.5
# Number of times to apply the blur
N_proj=1

#------------------- Domain Constants -------------------#
# Space between the PMLs and the design region (in pixels)
space=10
# Width of the waveguide (in pixels)
wg_width=12
# Length in pixels of the source/probe slices on each side of the center point
space_slice=8
# Distane between dots
dot_space=2
# Number of dots in a row/column
num_dots=5
# Radius of dots
circle_rad=int((Nx-2*(Npml+space))/(2*num_dots))

#---------------- Optimization Constants ----------------#
# Number of epochs in the optimization 
Nsteps=500
# Step size for the Adam optimizer
step_size=1e-3


def init_design(epsrs, Nx=Nx, Ny=Ny, Npml=Npml, space=space, dot_space=dot_space, num_dots=num_dots, circle_rad=circle_rad):
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
            r, c = circle(center_x, center_y, circle_rad)
            rods[j + i*num_dots][r, c] = epsrs[i, j]
            center_y = center_y + 2*circle_rad
        center_x = center_x + 2*circle_rad
        
    return rods, design_region
    
def init_guides(Nx=Nx, Ny=Ny, Npml=Npml, space=space, wg_width=wg_width, space_slice=space_slice, wg_shift=9):
    """Initializes waveguides and sources

    space       : The space between the PML and the structure
    wg_width    : The feed and probe waveguide width
    space_slice : The added space for the probe and source slices
    """
    bg_rho = np.zeros((Nx, Ny))
        
    # Input waveguide
    bg_rho[0:int(Npml+space),Ny//2-wg_width//2:Ny//2+wg_width//2] = 1
    # Input probe slice
    input_slice = Slice(x=np.array(Npml+1),
                        y=np.arange(Ny//2-wg_width//2-space_slice, Ny//2+wg_width//2+space_slice))    
    
    # Output waveguide
    bg_rho[int(Nx-Npml-space)::,Ny//2-wg_width//2:Ny//2+wg_width//2] = 1
    # Output probe slice
    output_slice = Slice(x=np.array(Nx-Npml-1),
                         y=np.arange(Ny//2-wg_width//2-space_slice, Ny//2+wg_width//2+space_slice))  
    
    # Ground waveguide
    bg_rho[Nx//2-wg_width//2:Nx//2+wg_width//2,int(Ny-Npml-space)::] = 1
    #Ground probe slice
    ground_slice = Slice(x=np.arange(Ny//2-wg_width//2-space_slice, Ny//2+wg_width//2+space_slice),
                         y=np.array(Ny-Npml-1))
            
    return bg_rho, input_slice, output_slice, ground_slice

def init_domain(epsrs):
    rods, design_region = init_design(epsrs)
    bg_rho, input_slice, output_slice, ground_slice = \
        init_guides(Nx, Ny, Npml, space=space, wg_width=wg_width, space_slice=space_slice, wg_shift=9)
    return bg_rho, design_region, rods, input_slice, output_slice, ground_slice


def viz_sim(epsr, source, slices=[]):
    """Solve and visualize a simulation with permittivity 'epsr'
    """   
    simulation = fdfd_ez(omega1, dl, epsr, [Npml, Npml])
    _, _, Ez = simulation.solve(source)
    
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(9,3))
    ceviche.viz.abs(Ez, outline=epsr, ax=axs[0], cbar=False)
    ceviche.viz.abs(epsr, ax=axs[1], cmap='Greys', cbar=True)
    for sl in slices:
        for ax in axs:
            try:
                ax.plot(sl.x*np.ones(len(sl.y)), sl.y, 'w-', alpha=0.5)
            except:
                ax.plot(sl.x, sl.y*np.ones(len(sl.x)), 'w-', alpha=0.5)
    
    fig.suptitle('$\lambda$ = %.2f $\mu$m' % (299792458/(omega1/2/np.pi)/1e-6))
    return (simulation, axs, fig)

def mask_combine_rho(rho, bg_rho, design_region):
    """Utility function for combining the design region rho and the background rho
    """
    return rho*design_region + bg_rho*(design_region==0).astype(np.float)

def scale_epsrs(epsrs, rods):
    epsrs = epsrs.flatten()
    rho = np.zeros(rods[0].shape)
    for i in range(len(epsrs)):
        rho += epsrs[i]*rods[i]
    return rho

def epsr_parameterization(epsrs, bg_rho, design_region, rods):
    """Defines the parameterization steps for constructing rho
    """
    rho = scale_epsrs(epsrs, rods)
    rho = mask_combine_rho(rho, bg_rho, design_region)
    
    return epsr_min + (epsr_max-epsr_min) * rho


# Initialize the parametrization rho and the design region
epsrs = np.ones((num_dots, num_dots))*0.5
bg_rho, design_region, rods, input_slice, output_slice, ground_slice = init_domain(epsrs)

# Compute the permittivity from the design_region and the plasma rod permittivities
epsr_init = epsr_parameterization(epsrs, bg_rho, design_region, rods)

# Setup sources
source = insert_mode(omega1, dl, input_slice.x, input_slice.y, epsr_init, m=1)
probe = insert_mode(omega1, dl, output_slice.x, output_slice.y, epsr_init, m=1)
ground = insert_mode(omega1, dl, ground_slice.x, ground_slice.y, epsr_init, m=1)

# Simulate initial device
simulation, ax, fig = viz_sim(epsr_init, source, slices = [input_slice, output_slice, ground_slice])


def callback_output_structure(iteration, of_list, epsrs):
    """Callback function to output fields and the structures (for making sweet gifs)"""
    epsrs = epsrs.reshape((num_dots, num_dots))
    epsr = epsr_parameterization(epsrs, bg_rho, design_region, rods)
    _, axs, fig = viz_sim(epsr, source, slices = [input_slice, output_slice])
    for ax in axs:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_xticks([])
    
    # note: make sure `workshop-invdesign/tmp` directory exists for this to work
    plt.savefig('tmp/epsr_%03d.png' % iteration, dpi=70)
    plt.close()

# Define optimization objective
def mode_overlap(E1, E2):
    """Defines an overlap integral between the sim field and desired field
    """
    return npa.abs(npa.sum(npa.conj(E1)*E2))*1e6

_, _, Ez00 = simulation.solve(source)
E00 = mode_overlap(Ez00, probe)
E11 = mode_overlap(Ez00, ground)


def objective1(epsrs):
    """Objective function called by optimizer
    
    1) Takes the density distribution as input
    2) Constructs epsr
    2) Runs the simulation
    3) Returns the overlap integral between the output wg field 
       and the desired mode field
    """
    epsrs = epsrs.reshape((num_dots, num_dots))
    epsr = epsr_parameterization(epsrs, bg_rho, design_region, rods)
    simulation.eps_r = epsr
    
    _, _, Ez00 = simulation.solve(source)
    return mode_overlap(Ez00, probe) / E00

print('Simulating Straight')
# Compute the gradient of the objective function using revere-mode differentiation
objective1_jac = jacobian(objective1, mode='reverse')
# Maximize the objective function using an ADAM optimizer
(epsrs_optimum, loss) = adam_optimize(objective1, epsrs.flatten(), objective1_jac,
                                      Nsteps=Nsteps, direction='max', step_size=step_size, callback=callback_output_structure)
np.savetxt('straight.csv', epsrs_optimum, delimiter=',')
make_gif('straight.gif')

def objective2(epsrs):
    """Objective function called by optimizer
    
    1) Takes the density distribution as input
    2) Constructs epsr
    2) Runs the simulation
    3) Returns the overlap integral between the output wg field 
       and the desired mode field
    """
    epsrs = epsrs.reshape((num_dots, num_dots))
    epsr = epsr_parameterization(epsrs, bg_rho, design_region, rods)
    simulation.eps_r = epsr
    
    _, _, Ez11 = simulation.solve(source)
    return mode_overlap(Ez11, ground) / E11

print('Simulating Bend')
# Compute the gradient of the objective function using revere-mode differentiation
objective2_jac = jacobian(objective2, mode='reverse')
# Maximize the objective function using an ADAM optimizer
(epsrs_optimum, loss) = adam_optimize(objective2, epsrs.flatten(), objective2_jac,
                                      Nsteps=Nsteps, direction='max', step_size=step_size, callback=callback_output_structure)
np.savetxt('bent.csv', epsrs_optimum, delimiter=',')
make_gif('bent.gif')