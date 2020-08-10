import numpy as np
import autograd.numpy as npa
import ceviche
from ceviche import fdfd_hz, fdfd_ez
import consts


def callback_output_structure(iteration, of_list, epsrs):
    """Callback function to output fields and the structures (for making sweet gifs)"""
    epsrs = epsrs.reshape((consts.num_dots, consts.num_dots))
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

    for i in range(len(bits)):
        if bits[i] == 1:
            desired.append(probe)
            undesired.append(ground)
            normal = normal * (mode_overlap(Ex_0[i], probe) / mode_overlap(Ex_0[i], ground))
        else:
            desired.append(ground)
            undesired.append(probe)
            normal = normal * (mode_overlap(Ex_0[i], ground) / mode_overlap(Ex_0[i], probe))

    def objective(epsrs):
        epsrs = epsrs.reshape((consts.num_dots, consts.num_dots))
        epsr = epsr_parameterization(epsrs, bg_rho, design_region, rods)
        simulation.eps_r = epsr
        
        Ex00, _, _ = simulation.solve(source)
        Ex01, _, _ = simulation.solve(source+dataB)
        Ex10, _, _ = simulation.solve(source+dataA)
        Ex11, _, _ = simulation.solve(source+dataA+dataB)
        Ex = [Ex00, Ex01, Ex10, Ex11]

        curr = np.prod([mode_overlap(Ex[i], desired[i]) / mode_overlap(Ex[i], undesired[i]) for i in range(len(bits))])        
        return curr / normal
    
    return objective

def simulate(bits):
    # Compute the gradient of the objective function using revere-mode differentiation
    objective = creat_objective(epsrs, bits)
    objective_jac = jacobian(objective, mode='reverse')

    # Maximize the objective function using an ADAM optimizer
    epsrs_optimum, _ = adam_optimize(objective, epsrs.flatten(), objective_jac,
                            Nsteps=Nsteps, direction='max', step_size=step_size, callback=callback_output_structure)
    return epsrs_optimum.reshape((consts.num_dots, consts.num_dots))