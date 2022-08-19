from re import L
import jax.numpy as jnp
from jax import lax.scan as scan
import matplotlib.pyplot as plt

def verlet_solver(rhs, ts, y0, v0):
    """Function to solve a system of two ODEs with Verlet integration. Algorithm is taken from
    url: https://physics.stackexchange.com/questions/239621/quadratic-drag-projectile-motion.
    
    args
    ----------
    rhs    callable
        Function which calculates accelerations. Call signature must be rhs(ys, vs), where
        ys and vs both have shape (n,). Must return array of shape (n,).
    ts      array
        Array of equidistant time points.
    y0      array
        Array of initial positions.
    v0      array
        Array of inital velocities.

    """
    dt = ts[1] - ts[0]  # step size
    # ys = jnp.zeros(shape=(ts.size, y0.size))  # initiate position array
    # vs = jnp.zeros(shape=(ts.size, v0.size))  # initiate velocity array

    # ys = ys.at[0].set(y0)
    # vs = vs.at[0].set(v0)
    # for i in range(len(ts) - 1):
    #     accs1 = rhs(ys[i], vs[i])
    #     v = vs[i] + 0.5 * accs1 * dt
    #     ys = ys.at[i + 1].set(ys[i] + v * dt)
    #     accs2 = rhs(ys[i + 1], v)
    #     vs = vs.at[i + 1].set(v + 0.5 * (accs2 - accs1) * dt)

    _, (x, v, theta, omega)

    return ys, vs


############## System ##############
def x_acc(force, theta, omega, alpha):
    return (force + m * l * (alpha * jnp.cos(theta) - omega**2 * jnp.sin(theta))) / (M + m)


def theta_acc(a, theta):
    return (g * jnp.sin(theta) + a * jnp.cos(theta)) / l


def get_acc(force, a, theta, omega, alpha):
    return x_acc(force, theta, omega, alpha), theta_acc(a, theta)


class Sys():
    def __init__(self, l, m, M):
        self.l = l
        self.m = m
        self.M = M
        
        
    def __repr__(self):
        return f'Sys(l={self.l}, m={self.m}, M={self.M})'
    
    
    def __str__(self):
        return f'Pendulum lenght is {self.l} m, pendulum mass is {self.m} kg, cart mass is {self.M} kg'


def rhs(ys, vs):
    return jnp.array([0, -1])
    

if __name__ == '__main__':
    ts = jnp.linspace(0, 6, 100)
    y0 = jnp.array([0, 0])
    v0 = jnp.array([jnp.sqrt(2), jnp.sqrt(2)])

    ys, vs = verlet_solver(rhs, ts, y0, v0)
    print(print(ys[ys[:,1] < 0.1]))
    plt.plot(ys[:, 0], ys[:, 1])
    plt.ylim(0,3)
    plt.show()

    exit()