## NumPy is used for randomly placing the agents on the lattice.
import numpy as np

## These are classes used to store the position of agents and the general image of the lattice.
from agent import Agent
from lattice import Lattice

class mc_Simulation:
    def __init__(self, size, num_agents, beta, sigma, gamma, p_exposed):
        ## Creates an object of the Lattice class, with the size of the lattice being determined by the parameter 'size'.
        self.__lattice = Lattice(size)
        ## An empty list that will be used to store all of the Agent class objects created throuhghout the simulation.
        self.__agents = []

        ## Initial conditions of infection spread and development for the sim.
        self.__beta = beta
        self.__sigma = sigma
        self.__gamma = gamma

        ## Method called to fill the lattice with the declared number of agents. Uses the p_exposed value to determine the initial distribution of exposed agents, with the rest being susceptible.
        self.__populate_agents(num_agents, p_exposed)

        ## Initialises the 'history' dictionary used to store the number of agents in each state at each time step.
        self.__history = {"S": [], "E": [], "I": [], "R": []}
    
    ## Method called from the constructor method, used to fill the lattice at t = 0.
    def __populate_agents(self, num_agents, p_exp):
        for _ in range(num_agents):
            ## Loop to ensure that no agents are placed on the same coordinates.
            while True:
                x = np.random.randint(0, self.__lattice.size - 1)
                y = np.random.randint(0, self.__lattice.size - 1)

                if self.__lattice.is_empty(x, y):
                    break

            ## Statement to determine whether an agent is initially susceptible or exposed based on the value of p_exposed.
            if np.random.random() < p_exp:
                state = 2  # exposed
            else:
                state = 1  # susceptible

            ## Creates an object of the Agent class for each placed agent and is initialised with initial coords and state. Each agent is appended to the self.__agents list and the lattice is updated based on the position and state of the agent using the set_stae method from the Lattice class.
            agent = Agent(x, y, state)
            self.__agents.append(agent)
            self.__lattice.set_state(x, y, state)

    ## Method to perform a single time step in the simulation. Every step, each agent attempts to move and then updates their state based on the results from the check_infection and update_state methods.
    def step(self):
        ## Lists to store the agent objects that have an altered state this time step.
        new_exposed = []
        new_infected = []
        new_recovered = []
        
        ## Loops through every agent object on the lattice, firstly trying to move it to another location, then based on this new location and its state, it either checks whether it becomes exposed or if it develops to a further state.
        for agent in self.__agents:
            agent.attempt_move(self.__lattice)

            if agent.check_infection(self.__lattice, self.__beta):
                new_exposed.append(agent)
            else:
                result = agent.update_state(self.__sigma, self.__gamma)
                if result == 'I':
                    new_infected.append(agent)
                elif result == 'R':
                    new_recovered.append(agent)
        
        ## At the end of the time step, all agents affected by a state change are updated to the new state. This prevents the order of the agents list from affecting the spread in the simulation in a way that is not true to reality.
        for agent in new_exposed:
            agent.set_state(2)
            self.__lattice.set_state(agent.x, agent.y, 2)
        for agent in new_infected:
            agent.set_state(3)
            self.__lattice.set_state(agent.x, agent.y, 3)
        for agent in new_recovered:
            agent.set_state(4)
            self.__lattice.set_state(agent.x, agent.y, 4)

        ## Calls the record method to store a snapshot of this time step.
        self.record()

    ## Stores the number of agents in each state within a dictionary, that can be used to plot SEIR curves at the end of the simulation.
    def record(self):
        counts = {1:0, 2:0, 3:0, 4:0}

        for agent in self.__agents:
            counts[agent.state] += 1

        self.__history["S"].append(counts[1])
        self.__history["E"].append(counts[2])
        self.__history["I"].append(counts[3])
        self.__history["R"].append(counts[4])

    ## Method to run all of the time steps in the simulation via the step method, then return the history dictionary for analysis and plotting.
    def run(self, steps):
        for _ in range(steps):
            self.step()
        return self.__history
