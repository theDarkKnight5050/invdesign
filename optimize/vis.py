import numpy as np
import matplotlib as mpl
import ceviche
from ceviche import fdfd_hz, fdfd_ez
from ceviche.optimizers import adam_optimize
from PIL import Image
import glob

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

def make_gif(name, Nsteps, folder, skip=1):
    # Create the frames
    frames = []
    for i in range(1, Nsteps, skip):
        new_frame = Image.open('%s/normalized_%03d.png' % (folder, i))
        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(name, format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=10)

def foo():
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