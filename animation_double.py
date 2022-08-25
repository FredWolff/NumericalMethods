import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch, Circle
from matplotlib.animation import FuncAnimation


def animate_double_pendulum(ts, ys, length_bottom, length_top, rolling_ax=False, filename=None):
    xs = ys[:, 0]
    theta1 = ys[:, 1]
    theta2 = ys[:, 2]
    length_parallel = length_bottom + length_top

    ylims = (-1 - length_parallel, length_parallel + 1)
    offset = 5

    fig, ax = plt.subplots()
    ax.set_ylim(*ylims)
    ax.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    cart_height = 0.2
    cart_width = 0.4
    init_pos = jnp.array([-cart_width / 2 + xs[0], -cart_height / 2])
    cart = Rectangle(init_pos, cart_width, cart_height, fill=True, color='r')

    def cartesian_unit(theta):
        return jnp.array([jnp.sin(theta), jnp.cos(theta)])

    init_xyA = jnp.array([xs[0], cart_height / 2])
    # Bottom Pendulum
    init_xyB = length_bottom * cartesian_unit(theta1[0]) + init_xyA
    pendulum_bottom = ConnectionPatch(init_xyB, init_xyA, coordsA=ax.transData)
    bob_middle = Circle(init_xyB, radius=0.05, color='b')
    # Top Pendulum
    init_xyC = length_top * cartesian_unit(theta2[0]) + init_xyB
    pendulum_top = ConnectionPatch(init_xyC, init_xyB, coordsA=ax.transData)
    bob_top = Circle(init_xyC, radius=0.05, color='b')
    # Time Display
    timetext = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    def init():
        timetext.set_text('')
        ax.add_patch(cart)
        ax.add_patch(pendulum_bottom)
        ax.add_patch(pendulum_top)
        ax.add_patch(bob_middle)
        ax.add_patch(bob_top)
        if rolling_ax == True:
            xlims = (-offset - length_parallel, length_parallel + offset)
        else:
            xlims = (-offset - length_parallel + jnp.min(xs), length_parallel + offset + jnp.max(xs))
        ax.set_xlim(*xlims)

        return []

    def animate(i):
        timetext.set_text('t = {:.1f}'.format(ts[i]))
        if rolling_ax == True:
            center = sum(ax.get_xlim())/2
            if jnp.absolute(xs[i] - center) > length_parallel:
                dif_sign = jnp.sign(xs[i] - center)
                xlims = (xs[i] - (dif_sign + 1) * length_parallel - offset, xs[i] + offset - (dif_sign - 1) * length_parallel)
                ax.set_xlim(*xlims)
        new_pos = jnp.array([xs[i], 0]) + jnp.array([-cart_width / 2, -cart_height / 2])
        cart.set_xy(new_pos)
        # New Bottom Pendulum Pos
        new_xy2 = jnp.array([xs[i], cart_height / 2])
        new_xy1 = length_bottom * cartesian_unit(theta1[i]) + new_xy2
        pendulum_bottom.xy1 = new_xy1
        pendulum_bottom.xy2 = new_xy2
        bob_middle.center = new_xy1
        # New Top Pendulum Pos
        new_xy3 = length_top * cartesian_unit(theta2[i]) + new_xy1
        pendulum_top.xy1 = new_xy3
        pendulum_top.xy2 = new_xy1
        bob_top.center = new_xy3

        return []

    anim = FuncAnimation(fig, animate, frames=len(ts), init_func=init, interval=(ts[-1] - ts[0]) / len(ts) * 1000, blit=True, repeat=True)

    # potentially save a file anim.save

    return anim










if __name__ == '__main__':
    from rk45_solver import rk45

    def rhs(y, t, m, M, l, g=9.81):
        x = y[0]
        theta = y[1]
        v = y[2]
        omega = y[3]

        v_dot = (m * g * jnp.sin(theta) * jnp.cos(theta) - m * l * omega**2 * jnp.sin(theta)) / (M + m - m * jnp.cos(theta)**2)
        omega_dot = g * jnp.sin(theta) / l * (M + m * (1 - l / g) * omega **2) / (M + m * (1 - jnp.cos(theta)**2))

        return jnp.array([v, omega, v_dot, omega_dot])

    fps = 30
    length_parallel = 1.
    y0 = jnp.array([0, 0, 5 / 10, 5], dtype=jnp.float32)
    ts = jnp.linspace(0, 5, 5 * fps)

    rhs_ = lambda y, t: rhs(y, t, 1., 10., 1.)
    ys = rk45(rhs_, ts, y0, h0=0.001)

    # anim = animate_pendulum(ts, ys, 10)
    # print(type(anim))
    # plt.show()

    anim = animate_pendulum(ts, ys, 1)
    fig, ax = plt.subplots()







# [xs, thetas, vs, omegas] = odeint(lambda y, t: rhs(y, t, **kwargs), ts, y0, h0)

# def loss(F, y0):
#     ys = odeint(lambda y, t: rhs(y, t, m, M, l, g, F), [0, h], y0, h0)

#     return ys[1]