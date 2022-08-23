"""Jax-based Runge-Kutta 4(5) solver. Heavily inspired by 
https://github.com/google/jax/blob/f905d989c11fb8a096b3c09d7f62faca485e9017/jax/experimental/ode.py#L97."""


import jax
import jax.numpy as jnp
from functools import partial


coeffsk = jnp.array([
    jnp.zeros(7),
    [1/5, 1/5, 0, 0, 0, 0, 0],
    [3/10, 3/40, 9/40, 0, 0, 0, 0],
    [4/5, 44/45, -56/15, 32/9, 0, 0, 0],
    [8/9, 19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
    [1, 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
    [1, 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
])
coeffs4 = jnp.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
coeffs5 = jnp.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])


def compute_ks(rhs, state):
    """Computes Runge-Kutta coefficients."""

    t_prev, y_prev, h = state

    def loop_step(i, ks):
        y_val = y_prev + coeffsk[i,1:] @ ks[:-1] * h
        t_val = t_prev + coeffsk[i,0] * h
        k_next = rhs(y_val, t_val)
        return ks.at[i].set(k_next)

    ks = jnp.zeros(shape=(7, y_prev.size))
    k = jax.lax.fori_loop(0, 7, loop_step, ks)

    return k


def compute_tol(y_prev, atol, rtol):
    return atol + rtol * jnp.abs(jnp.min(y_prev))


def compute_h(h, err, tol):
    factor = 0.9 * jnp.abs(tol / err)**0.2
    hs = jnp.array([2, factor, 0.5]) * h
    conds = jnp.array([factor > 2., (2. >= factor) & (factor >= 0.5), 0.5 > factor])

    return jnp.select(conds, hs)


def rk45(rhs, ts, y0, h0, atol=1e-5, rtol=1e-5):

    def rk45_step(state):
        t_prev, y_prev, h = state
        ks = compute_ks(rhs, state)
        y4 = y_prev + coeffs4 @ ks * h
        y5 = y_prev + coeffs5 @ ks * h
        t = t_prev + h
        err = jnp.abs(jnp.max(y5 - y4))
        tol = compute_tol(y_prev, atol, rtol)
        err_ratio = err / tol
        h = compute_h(h, err, tol)

        return [t, y5, h], err_ratio

    def scan_fun(carry, t_grid):
        """Function to repeat len(ts) times."""

        def cond_fun(state):
            """Terminates while loop when return value is False."""
            t_prev, _, h = state
            return (t_prev < t_grid) & (h > 0)

        def body_fun(state):
            """Takes a Runge-Kutta step. Returns old state with
            smaller time step if error exceeds tolerance."""

            new, err_ratio = rk45_step(state)
            t_prev, y_prev, _ = state
            _, _, h = new
            old = [t_prev, y_prev, h]

            return list(map(partial(jnp.where, err_ratio < 1.), new, old))
            
        carry = jax.lax.while_loop(cond_fun, body_fun, carry)
        t_prev, y_prev, h = carry
        h = t_grid - t_prev
        state = t_prev, y_prev, h
        carry, _ = rk45_step(state)
        _, y, _ = carry

        return carry, y

    y0 = jnp.asarray(y0)
    init_carry = [ts[0], y0, h0]
    _, ys = jax.lax.scan(scan_fun, init_carry, ts[1:])

    return jnp.concatenate((y0[None], ys))



# Test script

if __name__ == '__main__':
    def rhs(y, t, m, M, l, g=9.81):
        x = y[0]
        theta = y[1]
        v = y[2]
        omega = y[3]

        v_dot = (m * g * jnp.sin(theta) * jnp.cos(theta) - m * l * omega**2 * jnp.sin(theta)) / (M + m - m * jnp.cos(theta)**2)
        omega_dot = g * jnp.sin(theta) / l * (M + m * (1 - l / g) * omega **2) / (M + m * (1 - jnp.cos(theta)**2))

        return jnp.array([v, omega, v_dot, omega_dot])

    y0 = jnp.array([0, jnp.pi / 2, 0, 0])
    ts = jnp.linspace(0, 10, 10_000_000)

    # def kin(ys):
    #     v = ys[]

    rhs_ = lambda y, t: rhs(y, t, 1, 10, 10)
    ys = rk45(rhs_, ts, y0, h0=0.001)
    import matplotlib.pyplot as plt
    plt.plot(ts, ys)
    plt.show()