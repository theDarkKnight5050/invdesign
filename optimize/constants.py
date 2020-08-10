import collections
import math

#----------------- Simulation Constants -----------------#
# Create a container for our slice coords to be used for sources and probes
Slice = collections.namedtuple('Slice', 'x y')
# The angular frequencies
omega1=2*math.pi*200e12
omega2=2*math.pi*230e12
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