import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pylab as p
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import sys

# Code to draw figures with Newton method (e.g. Manifolds)


# Parameters

d = 2           # Dimension of the space
R = 2           # Size of the plotting box

def M(x):
    """Equation of the figure (or manifold).
    Input:
    - x: Array of shape (d,) - Variable"""
    x = x.reshape(d,)
    if d == 2:
        x1, x2 = x[0], x[1]

        # Unit spheres
        # y = x1**2 + x2**2 - 1
        # y = x1 ** 100 + x2 ** 100 - 1
        # y = np.max([np.abs(x1), np.abs(x2)]) - 1
        # y = np.abs(x1) + np.abs(x2) - 1

        # Elliptic curves
        # y = x2**2 - x1**3 + 3*x1 - 1
        # y = x2**3 - x1**3 + 3*x1*x2

        # Quintic curves
        # y = 20*x2*(x1**2+x2**2-1)*(5*x1**2+x2**2-2) + 1
        # y = x1*(x1*x2**3-14*x1*x2+1) + x2*(x2**4+10*x1**4-6*x2**2+4)
        # y = (7*x2**3-6*x1**2*x2-8*x1**2+7*x2**2+4)*(10*x1**2+6*x2**2+4*x2-9)-1

        # Hamiltonian functions
        y = (1/2)*x2**2 + (1-np.cos(x1)) - 1

    if d == 3:
        x1, x2, x3 = x[0], x[1], x[2]

        # Unit spheres
        # y = x1 ** 2 + x2 ** 2 + x3 ** 2 - 1
        # y = np.abs(x1) ** 3 + np.abs(x2) ** 3 + np.abs(x3) ** 3 - 1
        # y = np.abs(x1) + np.abs(x2) + np.abs(x3) - 1
        # y = np.max([np.abs(x1), np.abs(x2), np.abs(x3)]) - 1
        # y = x1 ** 10 + x2 ** 10 + x3 ** 10 - 1

        # Conics intersection [Solution of Rigid Body system]
        # y = np.abs(x1 ** 2 + x2 ** 2 + x3 ** 2 - 1) + np.abs(x1 ** 2 / 1 + x2 ** 2 / 0.7 + x3 ** 2 / 1.3  - 1)

        # "Cubic" curves
        # y = x1 ** 2 + x2 ** 2 + x3 ** 2 + x1 ** 2 * x3 - x2 ** 2 * x3 - 1
        # y = x1*x2 + x2*x3 + x3*x1
        # y = x1 ** 2 + x2 ** 2 - x3 ** 2 + 1
        # y = x1 ** 3 + x2 ** 3 - x3 ** 3 - (x1 + x2 + x3) **3



    return y

def dM(x):
    """Differeential function of M w.r.t. x.
    Input:
    - x: Array of shape (d,) - Variable"""
    eta = 1e-4
    x = x.reshape(d,)
    D = np.zeros_like(x)
    for k in range(d):
        e = np.zeros_like(x)
        e[k] = 1
        D[k] = (M(x+eta*e)-M(x-eta*e))/(2*eta)
    return D

def Plot(N = 100, N_iter = 10, lim_box = False):
    """Plots the figure by using Newton method (Moore-Penrose inverse is used to replace Inverse).
    Input:
    - N: Int - Number of starting points. Default: 100;
    - N_iter: Int - Number of iterations of Newton method. Default: 10
    - lim_box: Boolean - Plots the figure between limits -R and R for each axis or not. Default: False"""
    X = np.random.uniform(low=-R, high=R, size=(d,N))
    Y = np.zeros_like(X)

    print("Point...")
    for n in range(N):
        nn = n + 1
        sys.stdout.write("\r%d " % nn + "/" + str(N))
        sys.stdout.flush()

        xx = X[:,n]
        for k in range(N_iter):
            DD = dM(xx)
            xx = xx - M(xx)*DD/np.linalg.norm(DD)**2
        Y[:,n] = xx

    if d == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(Y[0,:],Y[1,:], color = "green", s = 1)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        if lim_box == True:
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
        plt.grid()
        ax.set_aspect("equal")
        plt.show()

    if d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(Y[0,:], Y[1,:], Y[2,:], c="green")#, depthshade=0)#, cmap="rainbow")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_zlabel("$z$")
        if lim_box == True:
            ax.set_xlim(-R, R)
            ax.set_ylim(-R, R)
            ax.set_zlim(-R, R)
        plt.grid()
        ax.set_box_aspect([1, 1, 1])
        #ax.set_aspect("equal")
        plt.show()

    pass

