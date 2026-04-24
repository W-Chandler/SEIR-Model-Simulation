## File containing Agent class. Each agent has attributes of position and SEIR state, as well as methods used for movement and updating states.
import random
random.seed(100) ## Set random seed for reproducibility

class Agent:
    def __init__(self, x, y, state):
        ## x and y represent the agent's initial position on the lattice at the point of creation.
        self.__x = x 
        self.__y = y
        self.__state = state ## Attribute to store SEIR state: 1 = S, 2 = E, 3 = I, 4 = R.
    
    # Property getters
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def state(self):
        return self.__state
    
    ## Method to attempt to move the agent to a new position on the lattice based on a random direction choice and whether the new position is empty or not.
    def attempt_move(self, lattice):
        ## Choose random direction: up, down, left or right.
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        dx, dy = random.choice(directions)
        
        ## Creates a new position for the agent
        new_x = self.__x + dx
        new_y = self.__y + dy
        
        ## Checks if new position is empty
        if lattice.is_empty(new_x, new_y):
            ## Clears old position
            lattice.set_state(self.__x, self.__y, 0)
            
            ## Updates position
            self.__x = new_x
            self.__y = new_y
            
            ## Sets new position
            lattice.set_state(self.__x, self.__y, self.__state)
    
    ## Method to check whether the agent becomes exposed based on the number of infected nighbours and infection probablility (beta).
    def check_infection(self, lattice, beta):
        if self.__state != 1:  ## Only susceptible can be infected
            return
        
        ## Get neighboring positions using method from the lattice class.
        neighbours = lattice.get_neighbours(self.__x, self.__y)
        
        ## Sums the amount of neighbouring agnests that are infected.
        infected_neighbours = 0
        for nx, ny in neighbours:
            if lattice.get_state(nx, ny) == 3:
                infected_neighbours += 1
                
        ## Checks whether any neighbours are infected, if not then the method ends here.
        if infected_neighbours > 0:
            ## Calculates infection probability based on number of infected neighbours.
            prob = 1 - (1 - beta) ** infected_neighbours
            if random.random() < prob:
                ## Changes state of Agent to exposed if the random number is less than the calculated probability, and updates the lattice to reflect this change.
                self.__state = 2
                lattice.set_state(self.__x, self.__y, 2)

    
    ## Method that uses the seeded random python library with the values of sigma and gamma to determine whether the state of the agent should be altered.
    def update_state(self, lattice, sigma, gamma):
        ## If agent is expsed, the value of sigma is used to determine whether it becomes infected.
        if self.__state == 2:
            if random.random() < sigma: ## sigma <= 1 in the monte-carlo simulation, differing from part 1 where it is a rate used in the ODE solver.
                ## Changes state of agent to infected if determined by the random number and updates lattice.
                self.__state = 3
                lattice.set_state(self.__x, self.__y, 3)
                
        ## If agent is infected the value of gamma is used to determine whether it becomes recovered.
        elif self.__state == 3:
            if random.random() < gamma:
                self.__state = 4
                lattice.set_state(self.__x, self.__y, 4)