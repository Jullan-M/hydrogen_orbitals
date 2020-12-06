import numpy as np
from scipy.special import eval_genlaguerre, sph_harm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm as cm
from matplotlib import colors as colors
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{eufrak}')

a = 0.529e-10

def radial(r, n, l):
    p = 2*r/(n*a)
    lag_gen = eval_genlaguerre(n-l-1, 2*l+1, p)
    y = np.sqrt((2/(n*a))**3 * np.math.factorial(n-l-1) / (2*n*np.math.factorial(n+l))) * np.exp(-p/2) * p**l * lag_gen
    return y

def wave_func(r, theta, phi, nlm):
    # r : radius
    # theta : polar angle
    # phi : azimuthal angle
    n, l, m = nlm

    # Note! Polar and azimuthal angle is denoted as phi and theta, respectively, in the docs for sph_harm.
    return radial(r, n, l) * sph_harm(m, l, phi, theta)

def wave_func_cart(x, y, z, nlm):
    xy = x**2 + y**2
    r = np.sqrt(xy + z**2)
    theta = np.arctan2(np.sqrt(xy), z)
    phi = np.arctan2(y,x)
    return wave_func(r, theta, phi, nlm)

class Orbital:
    def __init__(self, n: int, l: int, m: int, x : np.array = np.linspace(-2.5e-9, 2.5e-9, 401), z: float=0):
        self.n, self.l, self.m = n, l, m
        self.x = x
        self.dx = x[1]-x[0]

        xy, yx = np.meshgrid(x, x, indexing='ij')

        self.psi = wave_func_cart(xy, yx, z, (n,l,m))
        self.prob = self.psi.real**2 + self.psi.imag**2

class Transition:
    def __init__(self, orb : Orbital, fps : int = 60):
        self.fps = fps
        self.orb = orb
        self.prob_t = []
        self.waits = []
        self.tot_frames = 0

    def transition(self, other, duration = 2):
        orb2 = None
        if type(other) == tuple:
            orb2 = Orbital(*other)
        elif type(other) == Orbital:
            orb2 = other

        frames = duration * self.fps
        t = np.linspace(0, np.pi/2, frames)
        c1 = np.cos(t)
        c2 = np.sin(t)
        superpos = np.einsum("i,kj->ikj", c1, self.orb.psi) + np.einsum("i,kj->ikj", c2, orb2.psi)
        prob_dens = superpos.real**2 + superpos.imag**2
        
        self.prob_t.extend(prob_dens)
        self.tot_frames += frames
        self.orb = orb2

    def wait(self, duration):
        frames = duration * self.fps
        
        self.prob_t.append(self.orb.prob) # Add one frame of data to the beginnning
        self.prob_t.extend((frames-1)*[[]]) # The rest are empty data
        self.waits.append((self.tot_frames, self.tot_frames + frames))
        self.tot_frames += frames
        

    def save(self):

        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9, True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.set_axis_off()
        ax.contourf(self.orb.x, self.orb.x, self.prob_t[0], cmap=cm.get_cmap("magma"), levels = np.linspace(0, self.prob_t[0].max()/8, 101))
        ax.text(1.5e-9, -2e-9, r"$@ \mathfrak{J}\textrm{ullan\_M}$", fontsize=20, c="white")

        def animate(i):
            if self.waits and i > self.waits[0][0] and i <= self.waits[0][1]:
                if i == self.waits[0][1]:
                    self.waits.pop(0)
                return
            ax.clear()
            ax.contourf(self.orb.x, self.orb.x, self.prob_t[i], cmap=cm.get_cmap("magma"), levels = np.linspace(0, self.prob_t[i].max()/8, 101))
            ax.text(1.5e-9, -2e-9, r"$@ \mathfrak{J}\textrm{ullan\_M}$", fontsize=20, c="white")

        writer = animation.FFMpegWriter(self.fps, 'libx264', bitrate=4000)
        anim = animation.FuncAnimation(fig, animate, repeat=False, frames=self.tot_frames, interval=1000 / self.fps, blit=False)
        anim.save("transition.mp4", writer=writer, dpi=120, progress_callback=lambda i, n: print(f'{i} out of {n}.', end="\r"))

class Orbital_3D:
    def __init__(self, n: int, l: int, m: int, x : np.array = np.linspace(-2e-9, 2e-9, 201)):
        self.n, self.l, self.m = n, l, m
        self.x = x
        self.dx = x[1]-x[0]

        self.xyz, self.yxz, self.zxy = np.meshgrid(x, x, x, indexing='ij')

        self.psi = wave_func_cart(self.xyz, self.yxz, self.zxy, (n,l,m))
        self.prob = self.psi.real**2 + self.psi.imag**2

        norm2 = np.sum(self.prob)
        self.prob = self.prob / norm2
        self.psi = self.psi / np.sqrt(norm2)

    def snapshot(self):
        # Initiate figure and gridspec
        fig = plt.figure()
        gs = fig.add_gridspec(3,4, left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        ax_3d = fig.add_subplot(gs[:, :3], projection='3d')
        axs = [fig.add_subplot(gs[k, 3]) for k in range(3)]
        
        # Aesthetic configuration
        ax_3d.set_facecolor('black') 
        ax_3d.grid(False) 
        ax_3d.w_xaxis.pane.fill = False
        ax_3d.w_yaxis.pane.fill = False
        ax_3d.w_zaxis.pane.fill = False

        ax_3d.set_title(r"$\left | n, l, m \right \rangle = \left |" + f"{self.n}, {self.l}, {self.m} " + r"\right \rangle$", y=0.9, color="w")
        ax_3d.set_xlim3d(self.x[0], self.x[-1])
        ax_3d.set_ylim3d(self.x[0], self.x[-1])
        ax_3d.set_zlim3d(self.x[0], self.x[-1])

        xi, yi, zi = self.xyz.ravel(), self.yxz.ravel(), self.zxy.ravel()

        # Obtain indices of randomly selected points, as specified by probability density.
        randices = np.random.choice(np.arange(xi.shape[0]), 2000, replace = False, p = self.prob.ravel())
        
        # Random positions:
        x_rand, y_rand, z_rand = xi[randices], yi[randices], zi[randices]        
        
        vmin, vmax = self.prob.min(), self.prob.max() / 6
        ax_3d.scatter(x_rand, y_rand, z_rand, c=self.prob.ravel()[randices], cmap=cm.get_cmap("magma"), vmin = vmin, vmax = vmax, alpha=0.25)
        
        levels = np.linspace(vmin, vmax, 101)
        for ax, coord, slic in zip(axs, ("$xy$-plane", "$xz$-plane", "$yz$-plane"), (self.prob[:, :, 100], self.prob[:, 100, :], self.prob[100, :, :])):
            ax.set_axis_off()
            ax.set_title(coord, y=0.025, color="w")
            ax.contourf(self.x, self.x, slic.T, cmap=cm.get_cmap("magma"), levels = levels)

        plt.show()



class Transition_3D:
    def __init__(self, orb : Orbital_3D, fps : int = 60):
        self.fps = fps
        self.orb = orb
        self.prob_t = []
        self.waits = []
        self.tot_frames = 0

    def transition(self, other, duration = 2):
        orb2 = None
        if type(other) == tuple:
            orb2 = Orbital_3D(*other)
        elif type(other) == Orbital:
            orb2 = other

        frames = duration * self.fps
        t = np.linspace(0, np.pi/2, frames)
        c1 = np.cos(t)
        c2 = np.sin(t)
        superpos = np.einsum("i,kjl->ikjl", c1, self.orb.psi) + np.einsum("i,kjl->ikjl", c2, orb2.psi)
        prob_dens = superpos.real**2 + superpos.imag**2
        
        self.prob_t.extend(prob_dens)
        self.tot_frames += frames
        self.orb = orb2

    def wait(self, duration):
        frames = duration * self.fps
        
        self.prob_t.extend(frames*[self.orb.prob])
        self.waits.append((self.tot_frames, self.tot_frames + frames))
        self.tot_frames += frames

    def save(self):
        orb = self.orb
        samples = 2000

        # Initiate figure and gridspec
        fig = plt.figure()
        gs = fig.add_gridspec(3,4, left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        ax_3d = fig.add_subplot(gs[:, :3], projection='3d')
        axs = [fig.add_subplot(gs[k, 3]) for k in range(3)]
        # Aesthetic configuration
        ax_3d.view_init(15, -45)
        ax_3d.set_facecolor('black') 
        ax_3d.w_xaxis.pane.fill = False
        ax_3d.w_yaxis.pane.fill = False
        ax_3d.w_zaxis.pane.fill = False

        def ax3d_update(ax_3d):
            # This has to be updated every frame on the animation since we clear the frame on each iteration.
            # Ugh...
            ax_3d.set_axis_off()
            
            # TODO: Have to make a function that updates |nlm> based on time.
            #ax_3d.set_title(r"$\left | n, l, m \right \rangle = \left |" + f"{orb.n}, {orb.l}, {orb.m} " + r"\right \rangle$", y=0.9, color="w")
            ax_3d.set_xlim3d(orb.x[0], orb.x[-1])
            ax_3d.set_ylim3d(orb.x[0], orb.x[-1])
            ax_3d.set_zlim3d(orb.x[0], orb.x[-1])
        
        ax3d_update(ax_3d)
        xi, yi, zi = orb.xyz.ravel(), orb.yxz.ravel(), orb.zxy.ravel()     
        
        vmin, vmax = self.prob_t[0].min(), self.prob_t[0].max() / 6
        levels = np.linspace(vmin, vmax, 101)
        for ax, coord, slic in zip(axs, ("$xy$-plane", "$xz$-plane", "$yz$-plane"), (self.prob_t[0][:, :, 100], self.prob_t[0][:, 100, :], self.prob_t[0][100, :, :])):
            ax.set_axis_off()
            ax.set_title(coord, y=0.025, color="w")
            ax.contourf(orb.x, orb.x, slic.T, cmap=cm.get_cmap("magma"), levels = levels)

        axs[2].text(1.5e-9, -1.5e-9, r"$@ \mathfrak{J}\textrm{ullan\_M}$", fontsize=16, c="white")

        def animate(i):
            ax_3d.clear()
            ax3d_update(ax_3d)
            # Obtain indices of randomly selected points, as specified by probability density.
            randices = np.random.choice(np.arange(xi.shape[0]), samples, replace = False, p = self.prob_t[i].ravel())
            
            # Random positions:
            x_rand, y_rand, z_rand = xi[randices], yi[randices], zi[randices]
            vmin, vmax = self.prob_t[i].min(), self.prob_t[i].max() / 6

            ax_3d.scatter(x_rand, y_rand, z_rand, s=16, c=self.prob_t[i].ravel()[randices], cmap=cm.get_cmap("magma"), vmin = vmin, vmax = vmax, alpha=0.3)
            ax_3d.text2D(0.05, 0.025, r"$@ \mathfrak{J}\textrm{ullan\_M}$", fontsize=16, c="white", transform=ax_3d.transAxes)
            
            if self.waits and i > self.waits[0][0] and i <= self.waits[0][1]:
                if i == self.waits[0][1]:
                    self.waits.pop(0)
            else:
                levels = np.linspace(vmin, vmax, 101)
                for ax, coord, slic in zip(axs, ("$xy$-plane", "$xz$-plane", "$yz$-plane") ,(self.prob_t[i][:, :, 100], self.prob_t[i][:, 100, :], self.prob_t[i][100, :, :])):
                    ax.clear()
                    ax.set_title(coord, y=0.025, color="w")
                    ax.contourf(orb.x, orb.x, slic.T, cmap=cm.get_cmap("magma"), levels = levels)
                

        writer = animation.FFMpegWriter(self.fps, 'libx264', bitrate=4000)
        anim = animation.FuncAnimation(fig, animate, repeat=False, frames=self.tot_frames, interval=1000 / self.fps, blit=False)
        anim.save("transition.mp4", writer=writer, dpi=225, progress_callback=lambda i, n: print(f'{i} out of {n}.', end="\r"))

orb = Orbital_3D(4,1,0)
#orb.snapshot()

tr_3d = Transition_3D(orb, fps=10)
tr_3d.transition((4,2,0))
tr_3d.save()


'''
orb = Orbital(5,0,0)

tr = Transition(orb)
tr.wait(1)
tr.transition((4,0,0))
tr.wait(1)
tr.transition((3,0,0))
tr.wait(1)
tr.transition((3,1,0))
tr.wait(1)
tr.transition((3,1,1))
tr.wait(1)
tr.transition((4,0,0))
tr.wait(1)
tr.transition((4,1,1))
tr.wait(1)
tr.transition((4,2,1))
tr.wait(1)
tr.transition((4,3,1))
tr.wait(1)
tr.transition((5,0,0))
tr.wait(1)
tr.transition((5,1,1))
tr.wait(1)
tr.transition((5,2,1))
tr.wait(1)
tr.transition((5,3,1))
tr.wait(1)
tr.transition((5,4,1))
tr.wait(1)
tr.transition((5,3,1))
tr.wait(1)
tr.transition((5,2,1))
tr.wait(1)
tr.transition((5,1,1))
tr.wait(1)
tr.transition((1,0,0))
tr.wait(1)
tr.save()
'''