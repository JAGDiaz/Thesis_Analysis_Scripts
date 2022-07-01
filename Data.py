"""
    Created by:
        Opal Issan

    Modified:
        17 Nov 2020 - Jay Lago
"""
import numpy as np
from scipy.integrate import solve_ivp

# ==============================================================================
# Function Implementations
# ==============================================================================
def dyn_sys_discrete(lhs, mu=-0.05, lam=-1):
    """ example 1:
    ODE =>
    dx1/dt = mu*x1
    dx2/dt = lam*(x2-x1^2)

    By default: mu =-0.05, and lambda = -1.
    """
    rhs = np.zeros(2)
    rhs[0] = mu * lhs[0]
    rhs[1] = lam * (lhs[1] - (lhs[0]) ** 2.)
    return rhs

def dyn_sys_pendulum(lhs):
    """ pendulum example:
    ODE =>
    dx1/dt = x2
    dx2/dt = -sin(x1)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -np.sin(lhs[0])
    return rhs

def dyn_sys_fluid(lhs, mu=0.1, omega=1, A=-0.1, lam=10):
    """fluid flow example:
    ODE =>
    dx1/dt = mu*x1 - omega*x2 + A*x1*x3
    dx2/dt = omega*x1 + mu*x2 + A*x2*x3
    dx3/dt = -lam(x3 - x1^2 - x2^2)
    """
    rhs = np.zeros(3)
    rhs[0] = mu * lhs[0] - omega * lhs[1] + A * lhs[0] * lhs[2]
    rhs[1] = omega * lhs[0] + mu * lhs[1] + A * lhs[1] * lhs[2]
    rhs[2] = -lam * (lhs[2] - lhs[0] ** 2 - lhs[1] ** 2)
    return rhs

def dyn_sys_kdv(lhs, a1=0, c=3):
    """ planar kdv:
    dx1/dt = x2
    dx2/dt = a1 + c*x1 - 3*x2^2
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = a1 + c*lhs[0] - 3*lhs[0]**2
    return rhs

def dyn_sys_duffing_driven(t, lhs, alpha=1, beta=5, delta=2e-2, gamma=8, omega=.5):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = x - x^3 - gamma*y + alpha*cos(omega*t)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -delta*lhs[1] - beta*lhs[0]**3 - \
        alpha*lhs[0] + gamma*np.cos(omega*t)
    return rhs

def dyn_sys_duffing(t, rhs):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = x - x^3
    """
    lhs = np.zeros(2)
    lhs[0] = rhs[1]
    lhs[1] = rhs[0] - rhs[0]**3
    return lhs

def dyn_sys_jerk_circuit(t, lhs, A=.6):
    """ Jerk Circuit:
    dx/dt = y
    dy/dt = z
    dz/dt = -Az - y + |x| - 1
    """
    rhs = np.zeros(3)
    
    rhs[0] = lhs[1]
    rhs[1] = lhs[2]
    rhs[2] = abs(lhs[0]) - lhs[1] - A*lhs[2] - 1

    return rhs

def dyn_sys_lotka_volterra(t, rhs, alpha=2.5, beta=3, gamma=-1, delta=6):
    """ Lotka-Volterra:
    dx/dt = alpha*x - beta*x*y
    dy/dt = delta*x*y - gamma*y
    """
    lhs0 = alpha*rhs[0] - beta*rhs[0]*rhs[1]
    lhs1 = delta*rhs[0]*rhs[1] - gamma*rhs[1]
    return np.array([lhs0, lhs1])

def dyn_sys_VdP(t, rhs, mu=1.5):
    """ Van Der Pol Oscillator
    dx/dt = y
    dy/dt = mu*(1 - x^2)*y - x
    """
    lhs = np.zeros(2)
    lhs[0] = rhs[0]
    lhs[1] = mu*(1 - rhs[0]**2)*rhs[1] - rhs[0]
    return lhs

def dyn_sys_duffing_bollt(t, lhs, alpha=1.0, beta=-1.0, delta=0.5):
    """ Duffing oscillator:
    dx/dt = y
    dy/dt = -delta*y - x*(beta + alpha*x^2)
    """
    rhs = np.zeros(2)
    rhs[0] = lhs[1]
    rhs[1] = -delta*lhs[1] - lhs[0]*(beta + alpha*lhs[0]**2)
    return rhs

def dyn_sys_lorenz_63(t, rhs, sigma=10, rho=28, beta=8/3):
    """ Lorenz 63:
    dx/dt = sigma*(y - x)
    dy/dt = x*(rho - z) - y
    dz/dt = xy - beta*z
    """
    lhs = np.zeros(3)
    lhs[0] = sigma*(rhs[1] - rhs[0])
    lhs[1] = rhs[0]*(rho - rhs[2]) - rhs[1]
    lhs[2] = rhs[0]*rhs[1] - beta*rhs[2]
    return lhs

def dyn_sys_chua_circuit(t, rhs, func, kay=1, alpha=9.35159085, beta=14.790319805, gamma=0.016073965):
    """ Chua circuit:
    dx/dt = kay*alpha*(y - x - func(x))
    dy/dt = kay*(x - y + z)
    dz/dt = kay*(-beta*y + gamma*z)
    """
    lhs = np.zeros(3)
    lhs[0] = alpha*(rhs[1] - rhs[0] - func(rhs[1]))
    lhs[1] = kay*(rhs[0] - rhs[1] + rhs[2])
    lhs[2] = kay*(-beta*rhs[1] - gamma*rhs[2])
    return lhs

def dyn_sys_henon(rhs, a=1.4, b=.3):
    """ Henon map:
    x_{n+1} = 1 - a*(x_n)^2 + y_n
    y_{n+1} = b*x_n
    """
    lhs0 = 1 - a*rhs[0]**2 + rhs[1]
    lhs1 = b*rhs[0]
    return np.array([lhs0, lhs1])

def dyn_sys_strange_map(rhs):
    """ Strange map I found on wikipedia:
    x_{n+1} = 4*x_n*(1-x_n)
    y_{n+1} = (x + y) mod 1
    """
    lhs0 = 4*rhs[0]*(1 - rhs[0])
    lhs1 = (rhs[0] + rhs[1]) % 1.
    return np.array([lhs0, lhs1])

def data_maker_continuous_n_d(bounds, func, n_ic=10000, dt=.01, tf=1, seed=None, args=None, method='RK45'):
    np.random.seed(seed=seed)

    conds_list = [np.random.uniform(*bound, n_ic) for bound in bounds]
    init_conds = np.column_stack(conds_list)

    data_mat = [solve_ivp(fun=func, t_span=(0, tf), y0=init_cond, t_eval=np.arange(0, tf+dt, dt), method=method, args=args).y 
        for init_cond in init_conds]

    return np.transpose(data_mat, axes=[0, 2, 1])

def data_maker_discrete_n_d(bounds, func, n_ic=10000, iterations=100, seed=None, args=None):
    np.random.seed(seed=seed)

    conds_list = [np.random.uniform(*bound, n_ic) for bound in bounds]
    init_conds = np.row_stack(conds_list)

    states = [init_conds]*(iterations + 1)
    for ii in range(1, iterations + 1):
        states[ii] = func(states[ii - 1])

    return np.transpose(states, axes=[2, 0, 1])

# ==============================================================================
# Test program
# ==============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = True

    create_driven_duffing = False
    create_duffing_bollt = False
    create_lorenz = False
    create_jerk_circuit = False
    create_chua_circuit = False
    create_henon = False
    create_strange = False
    create_lotka_volterra = False
    create_VdP = True


    if create_driven_duffing:
        # Generate the data
        boundaries = [(-1, 1)]*2
        data = data_maker_continuous_n_d(boundaries, dyn_sys_duffing_driven, n_ic=5, dt=.05, tf=200)
    
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for datum in data:
            plt.plot(*datum.T)
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18)
        plt.title("Driven Duffing oscillator", fontsize=18)

    if create_VdP:
        # Generate the data
        boundaries = [(-2, 2)]*2
        data = data_maker_continuous_n_d(boundaries, dyn_sys_VdP, n_ic=5, dt=.02, tf=20)
    
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for datum in data:
            plt.plot(*datum.T)
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18)
        plt.title("Van Der Pol oscillator", fontsize=18)

    if create_lotka_volterra:
        # Generate the data
        boundaries = [(-1, 1)]*2
        data = data_maker_continuous_n_d(boundaries, dyn_sys_lotka_volterra, n_ic=1, dt=.05, tf=10, method='DOP853')
    
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for datum in data:
            plt.plot(*datum.T)
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18)
        plt.title("Lotka-Volterra", fontsize=18)

    if create_duffing_bollt:
        # Generate the data
        boundaries = [(-1, 1)]*2
        data = data_maker_continuous_n_d(boundaries, dyn_sys_duffing_bollt, n_ic=10, dt=.05, tf=200)
    
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for datum in data:
            plt.plot(*datum.T)
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18)
        plt.title("Duffing-Bollt oscillator", fontsize=18)

    if create_lorenz:
        dist = 20
        boundaries = [(-dist, dist)]*3

        data = data_maker_continuous_n_d(boundaries, dyn_sys_lorenz_63, n_ic=10, tf=50)

        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        for solution in data:
            ax.plot3D(*solution.T)
        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
        ax.set_title("Lorenz 63", fontsize=18)

    if create_jerk_circuit:
        dist = .5
        boundaries = [(-dist, dist)]*3
        data = data_maker_continuous_n_d(boundaries, dyn_sys_jerk_circuit, n_ic=5, dt=.1, tf=200)

        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        for solution in data:
            ax.plot3D(*solution.T)
        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$",
               xlim=(-2.5, 1.5), ylim=(-1.5, 2), zlim=(-1.5, 2))
        ax.set_title("Jerk Circuit", fontsize=18)

    if create_chua_circuit:
        dist = 1e-3
        boundaries = [(-dist, dist)]*3
        m0, m1 =  -2.7647222013,  0.1805569489
        func = lambda ex: m1*ex + .5*(m0-m1)*(abs(ex + 1) - abs(ex - 1))
        data = data_maker_continuous_n_d(boundaries, dyn_sys_chua_circuit, n_ic=100, dt=.05, tf=200, args=(func, 1.,  3.7091002664,  24.0799705758,  -0.8592556780,))

        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        for solution in data:
            ax.plot3D(*solution.T)
        ax.set(xlabel="$x$", ylabel="$y$", zlabel="$z$")
        ax.set_title("Chua Circuit", fontsize=18)

    if create_henon:
        # Generate the data
        dist = .1
        boundaries = [(-dist, dist)]*2
        data = data_maker_discrete_n_d(boundaries, dyn_sys_henon, n_ic=10, iterations=1000)
    
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for datum in data:
            plt.plot(*datum.T, 'o')
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18)
        plt.title("Henon Map", fontsize=18)

    if create_strange:
        # Generate the data
        dist = .1
        boundaries = [(-dist, dist)]*2
        data = data_maker_discrete_n_d(boundaries, dyn_sys_strange_map, n_ic=2, iterations=1000)
    
        # Visualize
        plt.figure(2, figsize=(8, 8))
        for datum in data:
            plt.plot(*datum.T, 'o')
        plt.xlabel("$x_1$", fontsize=18)
        plt.ylabel("$x_2$", fontsize=18)
        plt.title("Strange Map", fontsize=18)

    plt.show()
    print("done")
