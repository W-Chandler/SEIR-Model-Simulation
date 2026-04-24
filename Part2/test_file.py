## Testing file to check multiple areas of the simulation and ensure that everything is working as expected.
import numpy as np

from mc_simulation import mc_Simulation

## Test 1: Checking that agents don't move out of the boundaries.
def test_lattice_boundaries():
    np.random.seed(1)
    size = 10
    
    sim = mc_Simulation(size, num_agents = 1, beta = 0, sigma = 0, gamma = 0, p_exposed = 0)

    agent = sim.agents[0]

    for _ in range(100):
        agent.attempt_move(sim.lattice)

        assert 0 <= agent.x < size, "Boundary test failed: x out of bounds"
        assert 0 <= agent.y < size, "Boundary test failed: y out of bounds"

    print("Boundary test passed")


## Test 2: Checking that the agent can move to different locations on the lattice.
def test_agent_movement():
    np.random.seed(1)

    sim = mc_Simulation(size = 10, num_agents = 1, beta = 0, sigma = 0, gamma = 0, p_exposed = 0)

    lattice = sim.lattice
    agent = sim.agents[0]

    start_pos = (agent.x, agent.y)

    visited = [start_pos]
    moved = False
    # Run multiple movement attempts
    for _ in range(30):
        agent.attempt_move(lattice)
        visited.append((agent.x, agent.y))
        if (agent.x, agent.y) != start_pos:
            moved = True
    assert moved, "Agent did not move from starting position"

    print("Movement test passed")


## Test 3: Checking that the infection spreads correctly.
def test_infection_spread():
    np.random.seed(1)

    sim = mc_Simulation(size=5, num_agents=2, beta=1.0, sigma=0, gamma=0, p_exposed=0)

    agents = sim.agents

    ## Manually places the agents next to each other with determined states.
    agents[0]._set_agent(2, 2, sim.lattice, 1)
    agents[1]._set_agent(2, 3, sim.lattice, 3)

    ## Checks that the check_infection method correctly returns True for an agent that is about to become exposed.
    assert agents[0].check_infection(sim.lattice, 1), f'Infection did not spread as expected - state is {agents[0].state}'

    print("Infection test passed")


## Test 4: Checks that the state transitions from exposed to infected and infected to recovered work correctly.
def test_state_transitions():
    np.random.seed(1)

    sim = mc_Simulation(size=5, num_agents=2, beta=0, sigma=1.0, gamma=1.0, p_exposed=1.0)

    agents = sim.agents
    agents[0]._set_agent(0, 0, sim.lattice, 2)  ## exposed
    agents[1]._set_agent(0, 1, sim.lattice, 3)  ## infected

    sim.step()

    ## Checks that the exposed agent becomes infected and the infected agent recovers after a step with sigma = 1 and gamma = 1.
    assert agents[0].state == 3, f'State did not progress correctly - progressed to {agents[0].state} instead of 3'
    assert agents[1].state == 4, f'State did not progress correctly - progressed to {agents[1].state} instead of 4'

    print("State transition test passed")


## Test 5: Checks that the number of agents doesn't change during the simulation.
def test_conservation():
    np.random.seed(1)

    sim = mc_Simulation(size=10, num_agents=20, beta=0.2, sigma=0.2, gamma=0.2, p_exposed=0.2)

    initial_count = len(sim.agents)

    for _ in range(20):
        sim.step()

    final_count = len(sim.agents)

    assert initial_count == final_count, "Agent number changed"

    print("Conservation test passed")


## Test 6: Checks that the behaviour of the infection spread fits the expected trends for a high and low beta value.
## Gamma is set to 0.0 to prevent the number of infected reducing as this test is just looking for infection rates.
def test_outbreak_vs_decay():
    np.random.seed(1)

    ## High beta value
    sim_high = mc_Simulation(size = 10, num_agents = 50, beta = 1.0, sigma = 1.0, gamma = 0.0, p_exposed = 0.1)

    for _ in range(20):
        sim_high.step()

    ## Returns the final number of infected agents.
    I_high = sim_high.history["I"][-1]

    ## Low beta value
    sim_low = mc_Simulation(size = 10, num_agents = 50, beta = 0.01, sigma = 1.0, gamma = 0.0, p_exposed = 0.1)

    for _ in range(20):
        sim_low.step()

    I_low = sim_low.history["I"][-1]

    ## Checks that the final numberof infected in the outbreak scenario is larger than in the decya scenario, as expected.
    assert I_high >= I_low, "Outbreak behaviour not consistent"

    print("Outbreak/decay test passed")


## Main test runner.
if __name__ == "__main__":

    print("\nRunning Monte Carlo SEIR tests...\n")

    test_lattice_boundaries()
    test_agent_movement()
    test_infection_spread()
    test_state_transitions()
    test_conservation()
    test_outbreak_vs_decay()

    print("\nAll tests passed!\n")