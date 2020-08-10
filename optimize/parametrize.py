import numpy as np
import autograd.numpy as npa
import constants as consts

def mask_combine_rho(rods, bg_rho, design_region):
    """Utility function for combining the design region rho and the background rho
    """
    train = (consts.epsr_train_max-consts.epsr_train_min)*rods*(rods!=0).astype(np.float) + consts.epsr_train_min*(rods!=0).astype(np.float)
    design = consts.epsr_design*design_region*(rods==0).astype(np.float)
    bckgd = consts.epsr_bckgd*bg_rho*(design_region==0).astype(np.float)
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
