from re import L
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

g = 9.82

def verlet_solver(rhs, ts, x0, v0, system):
    """Function to solve a system of two ODEs with Verlet integration. Algorithm is taken from
    url: https://physics.stackexchange.com/questions/239621/quadratic-drag-projectile-motion.
    
    args
    ----------
    rhs    callable
        Function which calculates accelerations. Call signature must be rhs(ys, vs), where
        ys and vs both have shape (n,). Must return array of shape (n,).
    ts      array
        Array of equidistant time points.
    x0      array
        Array of initial positions.
    v0      array
        Array of inital velocities.
    system  class
        Class defining sizes in the system.

    """
    l, m, M = system.l, system.m, system.M
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

    def step(_, carry):  # could be vectorized in force for reinforcement learning
        """Function performing a single iteration in a velocity-Velvet solver
        using a predictor-corrector strategy to increase percision for a velocity-
        depedent acceleration. Return accomodates structure of jax.lax.scan.
        
        """
        force, p_vec, v_vec, a_vec = carry
        a_vec = get_acc(force, p_vec[1], v_vec[1], *a_vec)
        p_vec, v_vec = new_vec(p_vec, v_vec, a_vec)
        new_a_vec = get_acc(force, p_vec[1], v_vec[1], *a_vec)
        v_vec += dt * (new_a_vec - a_vec) / 2
        return (force, p_vec, v_vec, a_vec), (x, v, theta, omega)

    a0 = get_acc(force, x0[1], v0[1])
    init_val = (force, x0, v0, a0)
    _, (x, v, theta, omega) = jax.lax.scan(lambda c, x: step(x, c), init_val, ts)

    return 


############## System ##############
def theta_acc(A, force, omega, a, b, g = 9.82):
    return (g * b + A) / (l * (1 - m * a**2 / (m + M)))


def get_acc(force, theta, omega):
    a = jnp.cos(theta)
    b = jnp.sin(theta)
    m_tot = m + M
    A = (force - m * l * omega**2 * b) * a / m_tot
    x_acc = (
        (force + m * (g * b + A) * a / 
        (1 - m * a**2) / m_tot - 
        omega**2 * b) / m_tot
    ) 
    theta_acc = (
        (g * b + A) / (l * (1 - m * a**2 / m_tot))
    )
    return x_acc, theta_acc


def new_vec(p_vec, v_vec, a_vec):
    dpos = dt * (v_vec + dt * a_vec / 2)
    dvel = dt * a_vec
    return p_vec + dpos, v_vec + dvel


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
    ts = jnp.linspace(0, 3 * jnp.pi, 100)
    y0 = jnp.array([0, 0])
    v0 = jnp.array([jnp.sqrt(2), jnp.sqrt(2)])

    ys, vs = verlet_solver(rhs, ts, y0, v0)
    print(print(ys[ys[:,1] < 0.1]))
    plt.plot(ys[:, 0], ys[:, 1])
    plt.ylim(0,3)
    plt.show()

    exit()