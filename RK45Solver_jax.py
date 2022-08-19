from jax.experimental.ode import odeint
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

ts = jnp.linspace(0., 3 * jnp.pi, 10_000_000)
f0 = 10.
def rhs(f, t): return -f

t0 = time.perf_counter()
ys = odeint(rhs, f0, ts, rtol=1e-5, atol=1e-2)
t1 = time.perf_counter()
print(f'jax:\t{t1 - t0}')
print(ts.shape)
# plt.plot(ts, ys)
# plt.show()