import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.animation import FuncAnimation


def animate_pendulum2(ts, ys, length, filename=None):
    xs = ys[:, 0]
    thetas = ys[:, 1]

    xlims = (-2, 2)
    ylims = (-length, length)

    fig, ax = plt.subplots()
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    plt.gca().set_aspect('equal', adjustable='box')

    cart_height = 0.2
    cart_width = 0.4
    init_pos = jnp.array([-cart_width / 2 + xs[0], -cart_height / 2])
    cart = Rectangle(init_pos, cart_width, cart_height, fill=True, color='r')

    def cartesian_unit(theta):
        return jnp.array([-jnp.sin(theta), jnp.cos(theta)])

    init_xyB = jnp.array([xs[0], cart_height / 2])
    init_xyA = length * cartesian_unit(thetas[0]) + jnp.array([xs[0], -cart_height / 2])
    pendulum = ConnectionPatch(init_xyA, init_xyB, coordsA=ax.transData)
    

    def init():
        ax.add_patch(cart)
        ax.add_patch(pendulum)

        return []

    def animate(i):
        new_pos = jnp.array([xs[i + 1], 0]) + jnp.array([-cart_width / 2, -cart_height / 2])
        cart.set_xy(new_pos)
        new_xy2 = jnp.array([xs[i + 1], cart_height / 2])
        new_xy1 = length * cartesian_unit(thetas[i + 1]) + new_xy2
        pendulum.xy1 = new_xy1
        pendulum.xy2 = new_xy2

        return []

    anim = FuncAnimation(fig, animate, frames=len(ts) - 1, init_func=init, interval=ts[-1] / len(ts) * 1000, blit=True, repeat=False)
    plt.show()

    # potentially save a file anim.save

    return anim










if __name__ == '__main__':
    from NumericalMethods.rk45_solver import rk45

    def rhs(y, t, m, M, l, g=9.81):
        x = y[0]
        theta = y[1]
        v = y[2]
        omega = y[3]

        v_dot = (m * g * jnp.sin(theta) * jnp.cos(theta) - m * l * omega**2 * jnp.sin(theta)) / (M + m - m * jnp.cos(theta)**2)
        omega_dot = g * jnp.sin(theta) / l * (M + m * (1 - l / g) * omega **2) / (M + m * (1 - jnp.cos(theta)**2))

        return jnp.array([v, omega, v_dot, omega_dot])

    y0 = jnp.array([0, jnp.pi / 2, 0, 0])
    ts = jnp.linspace(0, 5, 5 * 30)

    rhs_ = lambda y, t: rhs(y, t, 0.1, 100, 1)
    ys = rk45(rhs_, ts, y0, h0=0.001)

    # anim = animate_pendulum(ts, ys, 10)
    # print(type(anim))
    # plt.show()

    anim, lens = animate_pendulum2(ts, ys, 1)
    fig, ax = plt.subplots()

    ax.plot(lens)







