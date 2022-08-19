import numpy as np
import matplotlib.pyplot as plt
import time

def rk45(F, t_span, f0, t_eval=None, h=0.01, atol=1e-2, rtol=1e-5):
    from array import array

    k_coeffs = np.array([
        np.zeros(7),
        [1/5, 1/5, 0, 0, 0, 0, 0],
        [3/10, 3/40, 9/40, 0, 0, 0, 0],
        [4/5, 44/45, -56/15, 32/9, 0, 0, 0],
        [8/9, 19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
        [1, 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
        [1, 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    ])
    f4_coeffs = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
    f5_coeffs = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    
    
    def compute_k(F, f_prev, t_prev, h):
        k = np.zeros(7)
        for i in range(7):
            k[i] = F(f_prev + k[:-1]@k_coeffs[i,1:]*h, t_prev + k_coeffs[i,0]*h)
        return k
        
        
    def compute_tol(f_prev, atol, rtol):
        return atol + rtol*np.abs(f_prev)
        
        
    def compute_h(f_prev, h, err, tol):
        with np.errstate(divide='raise'):
            try:
                factor = 0.9*np.abs(tol/err)**0.2
            except FloatingPointError:
                factor = 10
        if factor > 2:
            return 2*h
        elif factor < 0.5:
            return 0.5*h
        else:
            return factor*h
        
        
    if t_eval is None:
        def advance(F, f_prev, t_prev, h):
            k = compute_k(F, f_prev, t_prev, h)
            f4 = f_prev + f4_coeffs@k*h
            f5 = f_prev + f5_coeffs@k*h
            err = np.abs(f5 - f4)
            t = t_prev + h
            tol = compute_tol(f_prev, atol, rtol)
            h = compute_h(f_prev, h, err, tol)

            if err >= tol:
                return advance(F, f_prev, t_prev, h)
            else:
                return t, f5, h
    else:
        def advance(F, f_prev, t_prev, h, grid_index):
            grid_point = t_eval[grid_index]
            if t_prev + h > grid_point:
                h = grid_point - t_prev
                grid_index += 1
            k = compute_k(F, f_prev, t_prev, h)
            f4 = f_prev + f4_coeffs@k*h
            f5 = f_prev + f5_coeffs@k*h
            err = np.abs(f5 - f4)
            t = t_prev + h
            tol = compute_tol(f_prev, atol, rtol)
            h = compute_h(f_prev, h, err, tol)
            if err >= tol:
                return advance(F, f_prev, t_prev, h, grid_index)
            else:
                return t, f5, h, grid_index
        
        
    def solve(F, t_span, f0, h):
        t_start = t_span[0]
        t_stop = t_span[1]
        fs = array('d', [f0])
        ts = array('d', [t_start])
        i = 0
        grid_index = 1
        while ts[i] < t_stop:
            t, f, h, grid_index = advance(F, fs[i], ts[i], h, grid_index)
            fs.append(f)
            ts.append(t)
            i += 1
        fs = np.asarray(fs)
        ts = np.asarray(ts)
        return ts, fs
        
    ts, fs = solve(F, t_span, f0, h)
    return ts, fs
     

if __name__ == '__main__':
    ts = np.linspace(0., 3 * np.pi, 10_000_000)
    f0 = 10.
    def rhs(f, t): return -f
    t0 = time.perf_counter()
    ts, fs = rk45(rhs, t_span=(ts[0], ts[-1]), t_eval=ts, f0=f0, h=0.01)
    t1 = time.perf_counter()
    print(f'np:\t{t1 - t0}')
    plt.plot(ts, fs)
    plt.show()