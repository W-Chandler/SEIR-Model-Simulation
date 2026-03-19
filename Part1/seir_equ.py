"""
seir_equ.py

File to numerically calculate the SEIR model using the scipy package.
Produces a plot displaying the spread of infection over time using matplotlib.
"""

import numpy as np
from scipy.integrate import solve_ivp

def seir_odes(t, y, beta, sigma, gamma):
    ## t is not used within this function, but is required by solve_ivp. 
    s, e, i, r = y
    ## r is also not used in this function but is contained within y so is unpacked for clarity.
    ds_dt = -beta * i * s
    de_dt = (beta * i * s) - (sigma * e)
    di_dt = (sigma * e) - (gamma * i)
    dr_dt = gamma * i

    return [ds_dt, de_dt, di_dt, dr_dt]

def seir_solve(beta, sigma, gamma, s0, e0, i0, r0, t_end=100, t_steps=1000):
    t_eval = np.linspace(0, t_end, t_steps)
    t_span = (0, t_end)
    y0 = [s0, e0, i0, r0]
    
    ## dense_output = False tells the function to produce individual data points, not continuous
    solution = solve_ivp(
        fun = seir_odes,
        t_span = t_span,
        y0 = y0,
        method = 'RK45',
        args = (beta, sigma, gamma),
        t_eval = t_eval,
        dense_output = False)
    
    s, e, i, r = solution.y
    t = solution.t
    
    return t, s, e, i, r

## Function to check that the sum of s,e,i and r is equal to 1

def conservation(t, s,e,i,r):
    total = s + e + i + r
    tolerance = 1e-6
    values = np.abs(total - 1.0)
    worst_val = np.max(values)
    worst_conservation = t[np.argmax(values)]
    
    if worst_val < tolerance:
        print(f'Conservation check passed')
        return True
    else:
        print(f'Conservation check failed.')
        print(f'Max deviation: {worst_val:.6e}')
        print(f'Time at fail: {worst_conservation:.1f}')
        return False
        
    