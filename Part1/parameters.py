import os
import matplotlib.pyplot as plt
## Import the seir_solve function as the solver used for the test case is the same as what is required for future simulations.
from seir_equ import seir_solve

## Parent class of ParameterSweep used to contain the initial conditions and parameters of the simulation as private attributes, preventing accidental changes.
class SEIRSimulationBase:

    ## COnstructor method to initialise the initial attributes of the simulation.
    def __init__(self, beta, sigma, gamma, s0, e0, i0, r0, description):
        self.__beta  = beta
        self.__sigma = sigma
        self.__gamma = gamma
        self.__s0 = s0
        self.__e0 = e0
        self.__i0 = i0
        self.__r0 = r0
        self.__description = description
        self.__R0 = beta / gamma

########################################################
    ## Getter methods to retrieve the initial conditions if they are required outside of the SEIRSimulationBase class. Given the @property tag to make accessing them easier.
    @property
    def beta(self):
        return self.__beta

    @property
    def sigma(self):
        return self.__sigma

    @property
    def gamma(self):
        return self.__gamma

    @property
    def s0(self):
        return self.__s0

    @property
    def e0(self):
        return self.__e0

    @property
    def i0(self):
        return self.__i0

    @property
    def r0(self):
        return self.__r0

    @property
    def description(self):
        return self.__description

    @property
    def R0(self):
        return self.__R0
########################################################

## Class to run the simulation for given sets of parameters via methods from the seir_equ.py file. 
class ParameterSweep(SEIRSimulationBase):

    ## Constructor method which uses the parent class' constructor to assign inital conditions and parameters. Also initialises a result attribute as None so that it can be used to analyse the results of the simulation later.
    def __init__(self, beta, sigma, gamma, s0, e0, i0, r0, description):
        super().__init__(beta, sigma, gamma, s0, e0, i0, r0, description)
        self._result = None

    ## Method to call the seir_solve function, passing in the initial conditions.
    def run(self):
        self._result = seir_solve(
            beta = self.beta, sigma = self.sigma, gamma = self.gamma,
            s0 = self.s0, e0 = self.e0, i0 = self.i0, r0 = self.r0
        )
        
        ## Checks that the data used in the solver remains consistent with the total population and sends a warning if deviation is detected.
        if not self._result.conservation():
            print(f"Warning: conservation failed for {self.description}")

    ## Outputs a summary of the simulation results into the console.
    def print_summary(self):
        peak, day = self._result.peak_infected()
        print(f" R0: {self._result.R0:.2f}")
        print(f" Outbreak predicted: {self._result.outbreak()}")
        print(f" Peak infection: {peak:.3f} at day {day:.1f}")
        print(f" Final susceptible: {self._result.final_susceptible():.3f}")


    ## Method called from SweepCollection used to plot the result objects generated from ParameterSweep via the method plot_seir in seir_equ.py.
    def plot(self, ax, save_path = None):
        self._result.plot_seir(ax = ax, save_path = save_path)

## Class used to store multiple parameter sweeps, allowing for repeatable use of the simulation with varying paramters.
class SweepCollection:
    ## Creates an empty sweeps attribute used to store simulation results generated from the other classes.
    def __init__(self):
        self.__sweeps = []
    
    ## Method that adds a sweep to the collection, allowing for multiple sets of results to be stored and analysed.
    def add_sweep(self, sweep):
        self.__sweeps.append(sweep)
    
    ## Method to run all sweeps within the collection, using methods from the ParameterSweep class.
    def run_all(self):
        for sweep in self.__sweeps:
            sweep.run()
    
    ## Method to call the print_summary method from ParameterSweep for each set of results in the collection.
    def print_summary(self):
        if not self.__sweeps:
            print("No results available — run sweep first.")
            return
        
        print(f'{"="*50}\nSummary of all parameter sweeps\n{"="*50}')
        for sweep in self.__sweeps:
            print(f"\n{'-'*50}\n{ sweep.description }\n{'-'*50}")
            sweep.print_summary()
    
    ## Method to group all results into a single plot and save to the specified path.
    def plot_all(self, save_path = "Part1/figures/parameter_sweep.png"):
        n = len(self.__sweeps)
        ncols = 2
        nrows = (n + 1) // 2
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
        axes = axes.flatten()
        
        ## Loop used to plot each set of results form the simulation on a different subplot, using the description of each sweep as the title, along with the R0 value.
        for idx, sweep in enumerate(self.__sweeps):
            ax = axes[idx]
            ## Calls the plot method from ParameterSweep, which calls the plot_seir method from seir_equ.py to generate the SEIR curves or each set of results.
            sweep.plot(ax = ax)
            axes[idx].set_title(f'{sweep.description}')
        
        ## Loop to remove any empty sublots due to an odd number of sweeps being performed.
        for idx in range(n, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('SEIR Model Parameter Sweep', fontsize = 12, y = 1.02)
        plt.tight_layout()
        
        ## Allows the output graphs to be saved to the specified location, creating the directory if it doesn't exist prior to the program being ran.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.show()

## Function to get a user input and check whether it is in the given bounds.
def get_input(prompt, min_val, max_val):
    while True:
        try:
            val = float(input(prompt))
            if val < min_val or val > max_val:
                print(f"Value must be between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print("Invalid input, please enter a number.")

## Assigns values to the initial conditions via user input through the get_input function.
def get_initial_conditions():
    print("Enter initial conditions (values between 0 and 1, summing to 1):\n")
    while True:
        s0 = get_input(" Initial susceptible (s0): ", 0, 1)
        e0 = get_input(" Initial exposed (e0): ", 0, 1)
        i0 = get_input(" Initial infected (i0): ", 0, 1)
        r0 = get_input(" Initial recovered (r0): ", 0, 1)

        total = s0 + e0 + i0 + r0
        if abs(total - 1.0) < 1e-6:
            return s0, e0, i0, r0
        else:
            print("Values must sum to 1. Please re-enter the initial conditions.")


## Function to get values of parameters using the get_input function.
def get_parameters():
    print("Enter SEIR model parameters:\n")
    beta = get_input(" Transmission rate (β): ", 0, 10)
    sigma = get_input(" Incubation rate (σ): ", 0, 10)
    gamma = get_input(" Recovery rate (γ): ", 0, 10)

    return beta, sigma, gamma

## Main user input function of the program. Creates an object of SweepCollection and adds objects of the ParameterSweep class to it. Allows for multiple sets of variables to be added to the simulation until the user specifies otherwise.
def user_input():
    collection = SweepCollection()
    default_c = input("Use default conditions for all sets? (y/n): ").strip().lower()
    default_p = input("Use default parameters for all sets? (y/n): ").strip().lower()
    desc_c = input("Do you want the title's to display the initial conditions? (y/n): ").strip().lower()
    if desc_c != 'y':
        desc_p = input("Do you want the title's to display the parameters? (y/n): ").strip().lower()
    while True:
        if default_c == 'y':
            s0, e0, i0, r0 = 0.99, 0.01, 0, 0
        else:
            s0, e0, i0, r0 = get_initial_conditions()
        if default_p == 'y':
            beta, sigma, gamma = 1.0, 1.0, 0.1
        else:
            beta, sigma, gamma = get_parameters()
        if desc_c == 'y':
            description = f's0 = {s0}, e0 = {e0}, i0 = {i0}, r0 = {r0}'
        elif desc_p == 'y':
            description = f'β = {beta}, σ = {sigma}, γ = {gamma}, R0 = {beta/gamma:.1f}'
        
        sweep = ParameterSweep(s0 = s0, e0 = e0, i0 = i0, r0 = r0, 
                               beta = beta, sigma = sigma, gamma = gamma, 
                               description = description)
        collection.add_sweep(sweep)
        
        cont = input("Add another parameter set? (y/n): ").strip().lower()
        if cont != 'y':
            break
        
    return collection

## Execution block that is ran when the this file specifically is executed. Calls the user_input function to get parameters and conditions, then runs the parameter sets through the run_all function. Once the simulation is complete, a summary of each set is output and the plots are saved and displayed.
if __name__ == "__main__":
    collection = user_input()
    collection.run_all()
    collection.print_summary()
    collection.plot_all()