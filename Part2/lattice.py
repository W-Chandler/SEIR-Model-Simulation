## Import numpy library for the creation of the lattice grid.
import numpy as np

class Lattice:
    ## Constructor method to initialise the lattice with a given size.
    def __init__(self, size):
        self.__size = size
        self.__grid = np.zeros((size, size), dtype=int)
        
    #########################
    ## Property getters for if the values of size or grid are required outside of this class.
    @property
    def size(self):
        return self.__size
    
    @property
    def grid(self):
        return self.__grid

    ## Method to check if a position on the lattice is empty (state 0) and within bounds.
    def is_empty(self, x, y):
        if not self.in_bounds(x, y):
            return False
        return self.grid[y][x] == 0

    ## Method to alter the state of a position in the lattice.
    def set_state(self, x, y, state):
        if self.in_bounds(x, y):
            self.grid[y][x] = state

    ## Method to get the state of a position on the lattice, returning None if out of bounds.
    def get_state(self, x, y):
        if self.in_bounds(x, y):
            return self.grid[y][x]
        return None

    ## Method to check whether a position is in the bounds of the lattice.
    def in_bounds(self, x, y):
        return 0 <= x < self.__size and 0 <= y < self.__size

    ## Method to return the coordinates of the valid neighbouring positions.
    def get_neighbors(self, x, y):
        directions = [(0,1),(0,-1),(1,0),(-1,0)]
        neighbors = []

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            ## Checks if the new position is within bounds before adding to the list of neighbors.
            if self.in_bounds(nx, ny):
                neighbors.append((nx, ny))

        return neighbors