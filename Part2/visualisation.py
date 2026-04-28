import matplotlib.pyplot as plt
import numpy as np
from mc_simulation import mc_Simulation
import matplotlib.animation as animation
import matplotlib.patches as mpatches


class Visualisation:
    ## constructor class for the plotting and displaying of the simulation.
    def __init__(self, lattice, history, lattice_history):
        self.__lattice = lattice
        self.__history = history
        self.__lattice_history = lattice_history
        
        ## Derfines a colourmap for the different states of the agents, with the colour scheme matching the one used in the SEIR curves for consistncy.
        self.__colour_map = {
            0: [1, 1, 1],       # empty = white
            1: [0, 0, 1],       # S = blue
            2: [1, 0.5, 0],     # E = orange
            3: [0, 1, 0],       # I = green
            4: [1, 0, 0]        # R = red
        }

    ## Method to plot the final lattice state.
    def plot_lattice(self):
        ## Creates the lattice array using the lattic instance's grid property.
        grid = np.array(self.__lattice.grid)

        # Create empty RGB image to be filled by the various colours.
        rgb_grid = np.zeros((grid.shape[0], grid.shape[1], 3))

        ## Assigns colour values to the RGD grid.
        for state, colour in self.__colour_map.items():
            rgb_grid[grid == state] = colour

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(rgb_grid)
        ax.set_title("Final Lattice State")
        ax.invert_yaxis()

        # Legend setup.
        legend_handles = [
            mpatches.Patch(color = 'blue', label = 'Susceptible'),
            mpatches.Patch(color = 'orange', label = 'Exposed'),
            mpatches.Patch(color = 'green', label = 'Infected'),
            mpatches.Patch(color = 'red', label = 'Recovered')
        ]
        ax.legend(handles = legend_handles, loc = 'upper right')

        plt.show()

    ## Method to plot the SEIR curves using the dictionary 'history' that stores each step of the simulation.
    def plot_history(self):
        plt.figure(figsize=(8, 5))

        plt.plot(self.__history["S"], label = "Susceptible")
        plt.plot(self.__history["E"], label = "Exposed")
        plt.plot(self.__history["I"], label = "Infected")
        plt.plot(self.__history["R"], label = "Recovered")

        plt.xlabel("Monte Carlo Steps")
        plt.ylabel("Population")
        plt.title("SEIR Monte Carlo Simulation")

        plt.legend()
        plt.grid()

        plt.show()

    ## Method to output the lattice and SEIR curves together.
    def plot_all(self):
        self.plot_lattice()
        self.plot_history()
        self.animate_lattice()
    
    ## Method to create an animation of the lattice across all time steps in the simulation, using the lattice_history object.
    def animate_lattice(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ## Creates 2 attributes of the lattice, an initial state and a colour grid to match.
        grid1 = self.__lattice_history[0]
        rgb_grid1 = np.zeros((grid1.shape[0], grid1.shape[1], 3))

        for state, colour in self.__colour_map.items():
            rgb_grid1[grid1 == state] = colour

        im = ax.imshow(rgb_grid1)
        ax.set_xlim(0, grid1.shape[1])
        ax.set_ylim(0, grid1.shape[0])

        ## Update function to update each frame pof the animation.
        def update(frame):
            grid = self.__lattice_history[frame]

            rgb_grid = np.zeros((grid.shape[0], grid.shape[1], 3))

            for state, colour in self.__colour_map.items():
                rgb_grid[grid == state] = colour

            im.set_data(rgb_grid)
            ax.set_title(f"Step {frame}")
            
            return[im]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.__lattice_history),
            interval = 50,
            blit = False
        )

        plt.show()
        #ani.save("Simulation.gif", writer="pillow")
    
## Creates object of the mc_simulation class with the specified parameters.
sim = mc_Simulation(
    size = 40,
    num_agents = 20,
    beta = 0.4,
    sigma = 0.1,
    gamma = 0.005,
    p_exposed = 0.2
)

## Runs the simaulation for the specified number of steps.
results, lattice_history = sim.run(500)

## Creates an object of the visualisation class, passing in the lattice and lattice_history for the animation and plotting. Then uses the plot_all and animate methods to display the resultant images and animation.
viz = Visualisation(sim.lattice, results, lattice_history)
viz.plot_all()


print("Plotting finished")