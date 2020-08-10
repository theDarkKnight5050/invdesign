import sys, getopt, collections
import numpy as np
import autograd.numpy as npa
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt
from autograd.scipy.signal import convolve as conv
from skimage.draw import circle
import ceviche
from ceviche import fdfd_hz, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode
from make_gif import make_gif


#----------------- Simulation Constants -----------------#
# Create a container for our slice coords to be used for sources and probes
Slice = collections.namedtuple('Slice', 'x y')
# The angular frequencies
omega1=2*np.pi*200e12
# Wavevector
k0=omega1/299792458
# Spatial resolution in meters
dl=40e-9
# Number of pixels in x-direction
Nx=120
# Number of pixels in y-direction
Ny=120
# Number of pixels in the PMLs in each direction
Npml=20
# Minimum value of the relative permittivity
epsr_train_min=-12.0
epsr_train_max=1.0
# Maximum value of the relative permittivity
epsr_design=1.0
epsr_bckgd=12.0

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
num_dots=15
# Radius of dots
circle_rad=int((Nx-2*(Npml+space))/(2*num_dots))
# Lattice constant
ra=0.6

#---------------- Optimization Constants ----------------#
# Number of epochs in the optimization 
Nsteps=500
# Step size for the Adam optimizer
step_size=1e-3

#------------------ Runtime Constants -------------------#
# Detailed or consdensed output
skip=1
# Name
name="gate"
# Graphing
Emax=0
emin=0
emax=0


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvn:", ["name=", "bckgd=", "design=", "trnmin=", "trnmax="])
    except:
        print('Usage: plasma_rod_gates.py -v -n <Nsteps> --bckgd <eps background> --design <eps design region> --trnmin <eps training min> --trnmax <eps training min>')
        sys.exit(2)
    for opt, arg in opts:
        if opt =='-h':
            print('Usage: plasma_rod_gates.py -v -n <Nsteps> --bckgd <eps background> --design <eps design region> --trnmin <eps training min> --trnmax <eps training min>')
            sys.exit(0)
        elif opt == '-v':
            skip = 10
        elif opt == '-n':
            Nsteps = int(arg)
        elif opt == '--name':
            name = arg
        elif opt == '--bckgd':
            epsr_bckgd = int(arg)
        elif opt == '--design':
            epsr_design = int(arg)
        elif opt == '--trnmin':
            epsr_train_min = int(arg)
        elif opt == '--trnmax':
            epsr_train_max = int(arg)  

if __name__ == "__main__":
    main(sys.argv[1:])


def init_design(epsrs, Nx=Nx, Ny=Ny, Npml=Npml, space=space, num_dots=num_dots, circle_rad=circle_rad, ra=ra):
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
    
def init_guides(Nx=Nx, Ny=Ny, Npml=Npml, space=space, wg_width=wg_width, space_slice=space_slice, wg_shift=9):
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
    rods, design_region = init_design(epsrs)
    bg_rho, input_slices, output_slices = \
        init_guides(Nx, Ny, Npml, space=space, wg_width=wg_width, space_slice=space_slice, wg_shift=9)
    return bg_rho, design_region, rods, input_slices, output_slices


# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
def real(val, outline=None, ax=None, cbar=False, cmap='RdBu', outline_alpha=0.5, vmin=None, vmax=None):
    """Plots the real part of 'val', optionally overlaying an outline of 'outline'
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    
    if vmin is None:
        vmax = np.real(val).max()
        vmin = np.real(val).min()
    h = ax.imshow(np.real(val.T), cmap=cmap, origin='lower left', clim=(vmin, vmax), \
                  norm=MidpointNormalize(midpoint=0,vmin=vmin,vmax=vmax))
    
    if outline is not None:
        ax.contour(outline.T, 0, colors='k', alpha=outline_alpha)
    
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    if cbar:
        plt.colorbar(h, ax=ax)
    
    return ax

def abslt(val, outline=None, ax=None, cbar=False, cmap='magma', outline_alpha=0.5, outline_val=None, vmax=None):
    """Plots the absolute value of 'val', optionally overlaying an outline of 'outline'
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)      
    
    if vmax is None:
        vmax = np.abs(val).max()
    h = ax.imshow(np.abs(val.T), cmap=cmap, origin='lower left', vmin=0, vmax=vmax)
    
    if outline_val is None and outline is not None: outline_val = 0.5*(outline.min()+outline.max())
    if outline is not None:
        ax.contour(outline.T, [outline_val], colors='w', alpha=outline_alpha)
    
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    if cbar:
        plt.colorbar(h, ax=ax)
    
    return ax

def sim(epsr, source, dataA, dataB, slices=[], log=0):
    """Solve and visualize a simulation with permittivity 'epsr'
    """
    # a grave sin, but i've gotten lazy
    global emin, emax, Emax

    simulation = fdfd_hz(omega1, dl, epsr, [Npml, Npml])
    Ex00, _, _ = simulation.solve(source)
    Ex01, _, _ = simulation.solve(source+dataB)
    Ex10, _, _ = simulation.solve(source+dataA)
    Ex11, _, _ = simulation.solve(source+dataA+dataB)
    epsr[epsr >= epsr_train_max] = epsr_train_max

    if log:
        Estack = np.dstack((Ex00, Ex01, Ex10, Ex11))
        np.savetxt('tmp/Ex00_%03d.csv' % log, np.abs(Ex00), delimiter=',')
        np.savetxt('tmp/Ex01_%03d.csv' % log, np.abs(Ex01), delimiter=',')
        np.savetxt('tmp/Ex10_%03d.csv' % log, np.abs(Ex10), delimiter=',')
        np.savetxt('tmp/Ex11_%03d.csv' % log, np.abs(Ex11), delimiter=',')
        if np.abs(Estack).max() > Emax:
            Emax = np.abs(Estack).max()
        
        if epsr.min() < emin:
            emin=epsr.min()
        if epsr.max() > emax:
            emax=epsr.max()
        np.savetxt('tmp/epsr_%03d.csv' % log, epsr, delimiter=',')

    return simulation


def mask_combine_rho(rods, bg_rho, design_region):
    """Utility function for combining the design region rho and the background rho
    """
    train = (epsr_train_max-epsr_train_min)*rods*(rods!=0).astype(np.float) + epsr_train_min*(rods!=0).astype(np.float)
    design = epsr_design*design_region*(rods==0).astype(np.float)
    bckgd = epsr_bckgd*bg_rho*(design_region==0).astype(np.float)
    return train + design + bckgd

def scale_epsrs(epsrs, rods):
    epsrs = epsrs.flatten()
    epsrs = npa.arctan(epsrs) / np.pi + 0.5 
    rho = np.zeros(rods[0].shape)
    for i in range(len(epsrs)):
        rho += epsrs[i]*rods[i]
    return rho

def epsr_parameterization(epsrs, bg_rho, design_region, rods):
    """Defines the parameterization steps for constructing rho
    """
    rods = scale_epsrs(epsrs, rods)
    return mask_combine_rho(rods, bg_rho, design_region)


# Initialize the parametrization rho and the design region
epsrs = np.ones((num_dots, num_dots))*1.9
bg_rho, design_region, rods, input_slices, output_slices = init_domain(epsrs)

# Compute the permittivity from the design_region and the plasma rod permittivities
epsr_init = epsr_parameterization(epsrs, bg_rho, design_region, rods)

# Setup sources
source = insert_mode(omega1, dl, input_slices[0].x, input_slices[0].y, epsr_init, m=1)
dataA = insert_mode(omega1, dl, input_slices[1].x, input_slices[1].y, epsr_init, m=1)
dataB = insert_mode(omega1, dl, input_slices[2].x, input_slices[2].y, epsr_init, m=1)

# Setup probes
probe = insert_mode(omega1, dl, output_slices[0].x, output_slices[0].y, epsr_init, m=1)
ground = insert_mode(omega1, dl, output_slices[1].x, output_slices[1].y, epsr_init, m=1)

# Simulate initial device
simulation = sim(epsr_init, source, dataA, dataB, slices = input_slices + output_slices)


def callback_output_structure(iteration, of_list, epsrs):
    """Callback function to output fields and the structures (for making sweet gifs)"""
    epsrs = epsrs.reshape((num_dots, num_dots))
    epsr = epsr_parameterization(epsrs, bg_rho, design_region, rods)
    sim(epsr, source, dataA, dataB, slices = input_slices + output_slices, log=iteration)
    plt.close()

# Define optimization objective
def mode_overlap(E1, E2):
    """Defines an overlap integral between the sim field and desired field
    """
    return npa.abs(npa.sum(npa.conj(E1)*E2))*1e6

def creat_objective(epsrs, bits):
    """Objective function called by optimizer
    1) Takes the density distribution as input
    2) Constructs epsr
    2) Runs the simulation
    3) Returns the overlap integral between the output wg field 
       and the desired mode field
    """
    desired = []
    undesired = []
    normal = 1

    Ex00_0, _, _ = simulation.solve(source)
    Ex01_0, _, _ = simulation.solve(source+dataB)
    Ex10_0, _, _ = simulation.solve(source+dataA)
    Ex11_0, _, _ = simulation.solve(source+dataA+dataB)
    Ex_0 = [Ex00_0, Ex01_0, Ex10_0, Ex11_0]

    for i in range(1):#len(bits)):
        if bits[i] == 1:
            desired.append(probe)
            undesired.append(ground)
            normal = normal * (mode_overlap(Ex_0[i], probe) / mode_overlap(Ex_0[i], ground))
        else:
            desired.append(ground)
            undesired.append(probe)
            normal = normal * (mode_overlap(Ex_0[i], ground) / mode_overlap(Ex_0[i], probe))

    def objective(epsrs):
        epsrs = epsrs.reshape((num_dots, num_dots))
        epsr = epsr_parameterization(epsrs, bg_rho, design_region, rods)
        print(simulation.eps_r == epsr)
        simulation.eps_r = epsr
        
        Ex00, _, _ = simulation.solve(source)
        Ex01, _, _ = simulation.solve(source+dataB)
        Ex10, _, _ = simulation.solve(source+dataA)
        Ex11, _, _ = simulation.solve(source+dataA+dataB)
        Ex = [Ex00, Ex01, Ex10, Ex11]

        curr = np.prod([mode_overlap(Ex[i], desired[i]) / mode_overlap(Ex[i], undesired[i]) for i in range(1)])#   len(bits))])        
        return curr / normal
    
    return objective

def simulate(bits):
    # Compute the gradient of the objective function using revere-mode differentiation
    objective = creat_objective(epsrs, bits)
    objective_jac = jacobian(objective, mode='reverse')

    # Maximize the objective function using an ADAM optimizer
    epsrs_optimum, _ = adam_optimize(objective, epsrs.flatten(), objective_jac,
                            Nsteps=Nsteps, direction='max', step_size=step_size, callback=callback_output_structure)
    return epsrs_optimum.reshape((num_dots, num_dots))


# Simulate optimal device
np.savetxt('tmp/epsr_init.csv', simulate([0, 1, 1, 1]), delimiter=',')
obj1 = []
obj2 = []
obj3 = []
obj4 = []
for i in range(1, Nsteps):
    Ex00 = np.loadtxt('tmp/Ex00_%03d.csv' % i, delimiter=',')
    Ex01 = np.loadtxt('tmp/Ex01_%03d.csv' % i, delimiter=',')
    Ex10 = np.loadtxt('tmp/Ex10_%03d.csv' % i, delimiter=',')
    Ex11 = np.loadtxt('tmp/Ex11_%03d.csv' % i, delimiter=',')
    epsr = np.loadtxt('tmp/epsr_%03d.csv' % i, delimiter=',')
    obj1.append(mode_overlap(Ex00, probe))
    obj2.append(mode_overlap(Ex01, ground))
    obj3.append(mode_overlap(Ex10, probe) / mode_overlap(Ex10, ground))
    obj4.append(mode_overlap(Ex11, ground) / mode_overlap(Ex11, probe))
    
    fig, axs = plt.subplots(2, 5, constrained_layout=True, figsize=(16,16))
    epsr_graph = np.array(epsr, copy=True)
    epsr_graph[Npml+space:Nx-Npml-space, Npml+space:Ny-Npml-space] = 0
    
    abslt(Ex00, outline=epsr_graph, ax=axs[0,0], cbar=False, vmax=np.abs(Emax/10))
    axs[1,0].plot(obj1)
    abslt(Ex01, outline=epsr_graph, ax=axs[0,1], cbar=False, vmax=np.abs(Emax/10))
    axs[1,1].plot(obj2)
    abslt(Ex10, outline=epsr_graph, ax=axs[0,2], cbar=False, vmax=np.abs(Emax/10))
    axs[1,2].plot(obj3)
    abslt(Ex11, outline=epsr_graph, ax=axs[0,3], cbar=True, vmax=np.abs(Emax/10))
    axs[1,3].plot(obj4)
    real(epsr, ax=axs[0,4], cmap='RdGy', cbar=True, vmin=emin, vmax=emax)
    axs[-1][-1].axis('off')   

    plt.savefig('tmp/normalized_%03d.png' % i, dpi=70)
    plt.close()

make_gif('or_tm.gif', Nsteps, 'tmp', skip)