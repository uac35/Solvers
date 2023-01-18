#2D flow around a cylinder, using second order upwind
import numpy as np
import matplotlib.pyplot as plt

# Define the domain size and grid resolution
L = 1 # length of domain
N = 100 # number of grid points
dx = L/N # grid spacing

# Define the cylinder properties
R = 0.1 # radius of cylinder
xc = 0.5 # x-coordinate of cylinder center
yc = 0.5 # y-coordinate of cylinder center

# Define the flow properties
U = 1 # velocity in x-direction
V = 0 # velocity in y-direction

# Define the time step and number of time steps
dt = 0.01
n_steps = 1000

# Initialize the velocity and pressure fields
u = np.zeros((N,N))
v = np.zeros((N,N))
p = np.zeros((N,N))
x, y = np.meshgrid(np.linspace(0, L, N), np.linspace(0, L, N))


# Define the boundary conditions
u[:,0] = U
u[:,-1] = U
v[0,:] = V
v[-1,:] = V

# Iterate over time steps
for i in range(n_steps):
    # Compute the velocity at the cylinder surface
    u_cyl = U*(1-(R**2)/((x-xc)**2+(y-yc)**2))
    v_cyl = V*(1-(R**2)/((x-xc)**2+(y-yc)**2))

    # Implement the finite volume method with second-order upwind
    # discretization to update velocity and pressure fields
    for j in range(1,N-1):
        for k in range(1,N-1):
            u_star = u[j,k] - dt/dx*(u[j+1,k]*u[j+1,k] - u[j-1,k]*u[j-1,k])/2 + dt/dx*(v[j,k+1]*u[j,k+1] - v[j,k-1]*u[j,k-1])/2
            v_star = v[j,k] - dt/dx*(u[j+1,k]*v[j+1,k] - u[j-1,k]*v[j-1,k])/2 + dt/dx*(v[j,k+1]*v[j,k+1] - v[j,k-1]*v[j,k-1])/2
            u[j,k] = u_star
            v[j,k] = v_star
    # Apply boundary conditions
    u[:,0] = U
    u[:,-1] = U
    v[0,:] = V
    v[-1,:] = V

    # Plot the results in contours
    plt.contour(u)
    plt.hold(True)
    plt.contour(v)
    plt.hold(False)
    plt.show()
