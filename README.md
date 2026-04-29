# SEIR Epidemiological Model: ODE and Monte Carlo Simulation

This project implements a Susceptible-Exposed-Infected-Recovered (SEIR) model for epidemiological disease dynamics using two approaches: numerical ODE integration (Part 1) and spatial Monte Carlo simulation (Part 2).

## Project Overview

The SEIR model divides a population into four compartments:
- **S (Susceptible)**: Can contract the disease
- **E (Exposed)**: Infected but not yet infectious
- **I (Infected)**: Infectious and can spread disease
- **R (Recovered)**: Immune and cannot be reinfected

### Key Parameters
- **β (beta)**: Transmission rate — probability of infection upon contact with infected individuals
- **σ (sigma)**: Incubation rate — rate of progression from exposed to infected
- **γ (gamma)**: Recovery rate — rate of progression from infected to recovered
- **R₀**: Basic reproduction number = β/γ; indicates outbreak likelihood (R₀ > 1 predicts outbreak)

## Project Structure

```
Assessment4/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── report.tex               # LaTeX report (13 pages with figures and UML diagram)
│
├── Part1/                   # ODE-based SEIR solver
│   ├── seir_equ.py         # Numerical ODE integration using SciPy RK45
│   ├── parameters.py       # Parameter sweep framework
│   └── figures/            # Output plots (parameter sweeps, reference verification)
│
└── Part2/                   # Spatial Monte Carlo simulation
    ├── agent.py            # Individual agent class (position and SEIR state)
    ├── lattice.py          # 2D spatial grid for agent placement
    ├── mc_simulation.py     # Monte Carlo simulation orchestrator
    ├── visualisation.py     # Plotting and animation utilities
    ├── test_file.py        # Comprehensive unit tests
    └── figures/            # Output plots and animations
```

## Installation

### Prerequisites
- **Python 3.13+**
- **pip** (Python package manager)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

The project requires:
- **NumPy**: Numerical computations and random number generation
- **SciPy**: ODE integration (RK45 Dormand-Prince method)
- **Matplotlib**: 2D plotting and animation

## Usage

### Part 1: ODE-Based SEIR Solver

The ODE solver uses SciPy's adaptive step-size RK45 method to numerically integrate the SEIR differential equations.

#### Quick Start

Run the ODE solver with default parameters:

```bash
cd Part1
python seir_equ.py
```

**Expected Output:**
```
Running SEIR ODE solver with parameters:
  beta = 1.0, sigma = 1.0, gamma = 0.1
  s0 = 0.99, e0 = 0.01, i0 = 0.0, r0 = 0.0
  R0 = beta/gamma = 10.00

Conservation check passed
Peak infected: 0.549 at day 3.8
Remaining susceptible: 0.000
Outbreak predicted: True

[Plot displayed and saved to figures/reference_verification.png]
```

#### Parameter Sweep

Investigate effects of varying parameters using the sweep framework:

```bash
cd Part1
python parameters.py
```

This interactive script allows you to:
1. Set custom intial conditions
2. Set custom SEIR parameters (β, σ, γ)
3. Add multiple parameter sets for comparison
4. View consolidated plots with R₀ values

**Example Session:**
```
Enter initial conditions (values between 0 and 1, summing to 1):
 Initial susceptible (s0):
 Initial exposed (e0):
 Initial infected (i0):
 Initial recovered (r0):

Enter SEIR model parameters:
 Transmission rate (β): 0.1
 Incubation rate (σ): 1.0
 Recovery rate (γ): 0.1

Add another parameter set? (y/n): n
```

Plots are saved to `Part1/figures/parameter_sweep.png`.

#### Part 1 API Reference

**`seir_odes(t, y, beta, sigma, gamma)`**
- Core ODE system definition
- Returns: derivatives `[ds/dt, de/dt, di/dt, dr/dt]`

**`seir_solve(beta, sigma, gamma, s0, e0, i0, r0, t_end=100, t_steps=1000)`**
- Solves SEIR ODEs using RK45 adaptive step-size method
- Returns: `seir_results` object containing solutions and analysis methods

**`seir_results` Class**
- **Properties**: `R0`, `beta`, `sigma`, `gamma`, `t`, `s`, `e`, `i`, `r`
- **Methods**:
  - `outbreak()`: Returns `True` if R₀ > 1 (outbreak expected)
  - `peak_infected()`: Returns `(peak_value, day_at_peak)`
  - `final_susceptible()`: Returns proportion of uninfected at end
  - `conservation()`: Validates numerical accuracy (checks S+E+I+R ≈ 1)
  - `plot_seir(ax=None, save_path=None)`: Plots SEIR curves with optional save

**`ParameterSweep` Class**
- Runs ODE solver with specific parameters
- **Methods**: `run()`, `plot(ax, save_path)`, `print_summary()`

**`SweepCollection` Class**
- Aggregates multiple parameter sweeps
- **Methods**: `add_sweep()`, `run_all()`, `plot_all()`, `print_summary()`

---

### Part 2: Spatial Monte Carlo Simulation

The Monte Carlo simulation models spatial disease spread on a 2D lattice with individual agents exhibiting stochastic movement and state transitions.

#### Quick Start

Run the simulation with default parameters:

```bash
cd Part2
python visualisation.py
```

**Expected Output:**
- Lattice visualization showing final agent distribution (S=blue, E=orange, I=green, R=red)
- SEIR population time-series plot
- Animated GIF of lattice evolution (`Part2/figures/lattice_animation.gif`)

#### Key Simulation Parameters (in `visualisation.py`)

```python
sim = mc_Simulation(
    size=50,           # Lattice dimensions (50×50 grid)
    num_agents=300,    # Total number of agents
    beta=0.3,          # Infection probability per neighbor per step
    sigma=0.3,         # Probability of E→I transition
    gamma=0.005,       # Probability of I→R transition
    p_exposed=0.01     # Initial fraction of exposed agents
)

results, lattice_history = sim.run(100)  # Run 100 MC steps
```

#### Part 2 API Reference

**`Agent` Class**
- Encapsulates individual agent state and movement
- **Attributes** (read-only properties):
  - `x`, `y`: Current position on lattice
  - `state`: SEIR state (1=S, 2=E, 3=I, 4=R)
- **Methods**:
  - `attempt_move(lattice)`: Random movement to adjacent empty cell
  - `check_infection(lattice, beta)`: Determines S→E transition based on infected neighbors
  - `update_state(sigma, gamma)`: E→I and I→R transitions; returns transition type or None
  - `set_state(new_state)`: Updates agent's compartment
  - `_set_agent(x, y, lattice, state)`: Testing-only method for manual placement

**`Lattice` Class**
- 2D spatial grid managing agent positions
- **Attributes** (read-only properties):
  - `size`: Lattice dimension
  - `grid`: NumPy array of agent states at each location
- **Methods**:
  - `in_bounds(x, y)`: Validates coordinate within lattice
  - `is_empty(x, y)`: Checks if cell is vacant (state = 0)
  - `set_state(x, y, state)`: Updates cell value
  - `get_state(x, y)`: Reads cell value; returns None if out of bounds
  - `get_neighbours(x, y)`: Returns list of valid von Neumann neighbors (4-connected)

**`mc_Simulation` Class**
- Orchestrates Monte Carlo simulation steps
- **Attributes** (read-only properties):
  - `agents`: List of all Agent objects
  - `lattice`: Lattice object managing spatial layout
  - `history`: Dictionary with time-series of S, E, I, R counts
- **Methods**:
  - `step()`: Executes one MC time step (movement → infection → state transitions)
  - `run(steps)`: Executes `step()` for specified iterations; returns `(history, lattice_history)`
  - `record()`: Records current SEIR state distribution and lattice snapshot

**`Visualisation` Class**
- Generates plots and animations from simulation results
- **Methods**:
  - `plot_lattice()`: 2D grid visualization (color-coded by state)
  - `plot_history()`: Time-series SEIR plot; saves to `Part2/figures/seir_curves.png`
  - `animate_lattice()`: Creates and saves animation to `Part2/figures/lattice_animation.gif`
  - `plot_all()`: Combines all visualizations

#### Example: Custom Simulation

```python
from mc_simulation import mc_Simulation
from visualisation import Visualisation
import numpy as np

# Set seed for reproducibility
np.random.seed(100)

# Create and run simulation
sim = mc_Simulation(
    size=30,
    num_agents=200,
    beta=0.2,
    sigma=0.2,
    gamma=0.1,
    p_exposed=0.05
)

results, lattice_history = sim.run(200)

# Visualize
viz = Visualisation(sim.lattice, results, lattice_history)
viz.plot_history()
```

---

## Testing

The project includes comprehensive unit tests validating core functionality:

```bash
cd Part2
python test_file.py
```

**Tests Performed:**

| Test | Purpose |
|------|---------|
| `test_lattice_boundaries()` | Verifies agents cannot move outside lattice bounds |
| `test_agent_movement()` | Confirms agents can move to different positions |
| `test_infection_spread()` | Validates S→E transition when exposed to infected neighbor |
| `test_state_transitions()` | Confirms E→I and I→R transitions with sigma=1, gamma=1 |
| `test_conservation()` | Verifies agent count remains constant during simulation |
| `test_outbreak_vs_decay()` | Validates infection behavior differs for high vs. low beta |

**Expected Output:**
```
Running Monte Carlo SEIR tests...

Boundary test passed
Movement test passed
Infection test passed
State transition test passed
Conservation test passed
Outbreak/decay test passed

All tests passed!
```

---

## Parameter Investigation Results

### Part 1: ODE Parameter Sweep Findings

**Beta (Transmission Rate) Effects:**
- **β = 0.05**: R₀ = 0.5 → No outbreak, disease dies out
- **β = 0.1**: R₀ = 1.0 → Threshold case; outbreak just begins
- **β = 0.5**: R₀ = 5.0 → Strong outbreak; high peak infection
- **β = 1.0**: R₀ = 10.0 → Severe outbreak; rapid epidemic

**Critical Threshold**: R₀ = 1 (β = γ = 0.1) marks transition from disease decay to epidemic outbreak.

**Gamma (Recovery Rate) Effects:**
- Lower γ → Longer infectious period → Higher peak infections and longer epidemic duration
- Higher γ → Rapid recovery → Rapid epidemic termination with fewer total infections

**Initial Exposure Effects:**
- Varying e₀ (initial exposed fraction) primarily affects epidemic timing, not final outcome
- Higher initial exposure accelerates outbreak onset
- Total attack rate converges regardless of initial conditions for fixed β, σ, γ

### Part 2: Monte Carlo Spatial Dynamics

The Monte Carlo model reveals:
- Spatial structure suppresses infection spread compared to ODE model
- Local clustering of infected agents affects spread of infection
- Lattice dimension and agent density significantly affect outbreak dynamics
- Stochastic variability produces outcomes reliant on more elements than the ODE model

---

## Reproducibility

### Random Seed Configuration

Both modules use seeded random number generation for reproducibility:
- **Part 1**: `scipy.integrate.solve_ivp` uses adaptive stepping (deterministic for fixed parameters)
- **Part 2**: `numpy.random.seed(100)` set at module level in `agent.py` and `mc_simulation.py`

## Output Files

### Part 1 Outputs (`Part1/figures/`)
- `reference_verification.png`: SEIR curves for default parameters
- `parameter_sweep.png`: Multi-panel SEIR plots for parameter sweep investigation

### Part 2 Outputs (`Part2/figures/`)
- `seir_curves.png`: Population time-series from Monte Carlo run
- `lattice_animation.gif`: Frame-by-frame lattice evolution animation

---

## Key Findings Summary

1. **R₀ Threshold**: R₀ = 1 (equivalently, β = γ) marks definitive boundary between disease eradication and epidemic outbreak
2. **Spatial vs. Well-Mixed**: Monte Carlo lattice model suppresses infection compared to ODE well-mixed assumption; realistic infection spread requires accounting for spatial structure
3. **Parameter Sensitivity**: Gamma parameter critically impacts epidemic duration and final attack rate; beta parameter determines outbreak likelihood
4. **Stochastic Effects**: Individual-based simulation displays greater variance in outcomes than ODE.

---

