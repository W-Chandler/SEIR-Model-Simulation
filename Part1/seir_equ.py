import numpy as np
from scipy.integrate import solve_ivp

## Function to calculate SEIR equations.
def seir_odes(t, y, beta, sigma, gamma):
    ## t is not used within this function, but is required by solve_ivp. 
    s, e, i, r = y
    ## r is also not used in this function but is contained within y so is unpacked for clarity.
    ds_dt = -beta * i * s
    de_dt = (beta * i * s) - (sigma * e)
    di_dt = (sigma * e) - (gamma * i)
    dr_dt = gamma * i

    return [ds_dt, de_dt, di_dt, dr_dt]

## Produces data values to be plotted via the solve_ivp function.
def seir_solve(beta, sigma, gamma, s0, e0, i0, r0, t_end=100, t_steps=1000):
    t_eval = np.linspace(0, t_end, t_steps)
    t_span = (0, t_end)
    y0 = [s0, e0, i0, r0]
    
    ## Passes values of β, σ, and γ into solve_ivp as the args parameter, which are used in the seir_odes function to calculate the derivatives for each time step. The results of the integration are stored as variables s, e, i, r and t.
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

## Class structure to protect the attributes used, preventing accidental modification in further sections of the code.
class seir_results:
    ## Initialises the attributes of the class.
    def __init__(self, t, s, e, i, r, beta, sigma, gamma):
        self.t = t
        self.s = s
        self.e = e
        self.i = i
        self.r = r
        self.__beta = beta
        self.__sigma = sigma
        self.__gamma = gamma
        self.__R0 = beta / gamma
    
    ## Getter methods used to retrieve the private attributes if required outside of the seir_results class. Labelled with the tag @property to make accessing them easier.
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
    
    ## Function to quickly predict if an outbreak is expected using the value of R_0.
    def outbreak(self):
        return self.R0 > 1.0

    ## Function to find the day with the largest number of infected people and return the day and amount of infected.
    def peak_infected(self):
        peak = np.max(self.i)
        day  = self.t[np.argmax(self.i)]
        return peak, day

    ## Finds the remaining number of susceptible people at the end of the simulation.
    def final_susceptible(self):
        return self.s[-1]

    ## Method to check whether the initial conditions are appropriate for the simulation, and repeatedly check throughout that the solver's numerical drift is not significantly impacting the results by affecting the total population values.
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

    ## Method to plot the SEIR curves, using values from both the seir_equ.py file and the parameters.py file. If a save_path is provided, the resultant plot will be saved to that path.
    def plot_seir(self, ax = None, save_path = None):
        ## Checks whether the data is from seir_equ (ax is None) or parameters.
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            multiple = False
        else:
            multiple = True
        
        # Plot the curves with colours consistent with the given instruction set.
        ax.plot(self.t, self.s, label = 'Susceptible', color = 'blue')
        ax.plot(self.t, self.e, label = 'Exposed',     color = 'orange')
        ax.plot(self.t, self.i, label = 'Infected',    color = 'green') 
        ax.plot(self.t, self.r, label = 'Recovered',   color = 'red') 
        
        # Set labels and styling
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Fraction of population')
        ax.set_title(f'SEIR model with beta = {self.beta}, sigma = {self.sigma}, gamma = {self.gamma}')
        ax.legend()
        ax.grid(True)
        
        ## Outputs/saves the plot if only one plot is being produced.
        if not multiple:
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()
    
    ## Displays a summary of the results in the console.
    def print_summary(self):
        peak, day = self.peak_infected()
        print(f'Peak infected: {peak:.3f} at day {day:.1f}')
        print(f'Remaining susceptible: {self.final_susceptible():.3f}')
        print(f'Outbreak predicted: {self.outbreak()}')


## Runs the seir_solve function if this file is run as a script, using the test paramters given in the instructions.
if __name__ == "__main__":
    beta  = 1.0
    sigma = 1.0
    gamma = 0.1
    s0, e0, i0, r0 = 0.99, 0.01, 0.0, 0.0
    
    print("Running SEIR ODE solver with parameters:")
    print(f'  beta = {beta}, sigma = {sigma}, gamma = {gamma}')
    print(f'  s0 = {s0}, e0 = {e0}, i0 = {i0}, r0 = {r0}')
    print(f'  R0 = beta/gamma = {beta/gamma:.2f}')

    ## Runs the solver and stores results in an object of the seir_results class.
    result = seir_solve(
        beta = beta, sigma = sigma, gamma = gamma,
        s0 = s0, e0 = e0, i0 = i0, r0 = r0
    )
    
    ## Verifies that the total population is constant and prints a warning if it is not.
    if not result.conservation():
        print("Aborting - solver can't be trusted")
        exit(1)
    
    result.print_summary()
    
    ## Plots the SEIR curves and saves the resulatant file to the specified path.
    result.plot_seir(save_path = 'figures/reference_verification.png')