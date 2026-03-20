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
    
    return seir_results(t, s, e, i, r, beta, sigma, gamma)

import matplotlib.pyplot as plt
import os

class seir_results:
    def __init__(self, t, s, e, i, r, beta, sigma, gamma):
        self.t     = t
        self.s     = s
        self.e     = e
        self.i     = i
        self.r     = r
        self.__beta  = beta
        self.__sigma = sigma
        self.__gamma = gamma
        self.__R0    = beta / gamma
    
    @property
    def R0(self):
        return self.__R0

    @property
    def beta(self):
        return self.__beta

    @property
    def sigma(self):
        return self.__sigma

    @property
    def gamma(self):
        return self.__gamma
        
    def outbreak(self):
        return self.R0 > 1.0

    def peak_infected(self):
        peak = np.max(self.i)
        day  = self.t[np.argmax(self.i)]
        return peak, day

    def final_susceptible(self):
        return self.s[-1]

    def conservation(self):
        total = self.s + self.e + self.i + self.r
        tolerance = 1e-6
        values = np.abs(total - 1.0)
        worst_val = np.max(values)
        worst_conservation = self.t[np.argmax(values)]
        
        if worst_val < tolerance:
            print(f'Conservation check passed')
            return True
        else:
            print(f'Conservation check FAILED.')
            print(f'Max deviation: {worst_val:.2e}')
            print(f'Time at fail: {worst_conservation:.1f}')
            return False

    def plot_seir(self):
        save_path='figures/reference_verification.png'
        plt.figure(figsize=(10,6))
        
        plt.plot(self.t, self.s, label='Susceptible', color='blue')
        plt.plot(self.t, self.e, label='Exposed',     color='orange')
        plt.plot(self.t, self.i, label='Infected',    color='green')
        plt.plot(self.t, self.r, label='Recovered',   color='red')
        
        plt.xlabel('Time (days)')
        plt.ylabel('Fraction of population')
        plt.title(f'SEIR model with beta = {self.beta}, sigma = {self.sigma}, gamma = {self.gamma}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(save_path)
        
        plt.show()




if __name__ == "__main__":
    beta  = 1.0
    sigma = 1.0
    gamma = 0.1
    s0, e0, i0, r0 = 0.99, 0.01, 0.0, 0.0
    
    print("Running SEIR ODE solver with parameters:")
    print(f'  beta={beta}, sigma={sigma}, gamma={gamma}')
    print(f'  s0={s0}, e0={e0}, i0={i0}, r0={r0}')
    print(f'  R0 = beta/gamma = {beta/gamma:.2f}')

    result = seir_solve(
        beta=beta, sigma=sigma, gamma=gamma,
        s0=s0, e0=e0, i0=i0, r0=r0
    )
    
    ## Verifying solver
    if not result.conservation():
        print("Aborting - solver can't be trusted")
        exit(1)
    
    peak, day = result.peak_infected()
    print(f'Peak infected: {peak:.3f} at day {day:.1f}')
    print(f'Remaining susceptible: {result.final_susceptible():.3f}')
    print(f'Outbreak predicted: {result.outbreak()}')
    
    result.plot_seir()
    