# Based on model.py
import pyomo.environ as pyo
import pyomo.dae as dae

import params_initialization
from model.utils import solve_model
from model.pore_concentration_profile import add_bvp_constraints
from model.reactor_concentration_profile import add_reactor_odes
import model.visualization as viz

def build_reactor_model(sid='gamma', decay=False):
    model = pyo.ConcreteModel()
    
    # Model parameters indexing - one stage, 3 substrates
    model.Stage      = pyo.Set(initialize=[1])  # Single stage
    model.Components = pyo.Set(initialize=['S1', 'S2', 'S3']) # Substrate components
    
    # Load parameters
    model = params_initialization.load_parameters_alpha(model)  # load_parameters.py
    
    # Establish model time and space domains (independent vars.)
    model.time  = dae.ContinuousSet(bounds=(0,model.tf))     # Reaction time
    model.x     = dae.ContinuousSet(bounds=(0, model.L))     # Pore x-spatial dimension
    
    # State variables (for IVP) index order:-> Components, time
    model.S_0 = pyo.Var(model.Components, model.time) # Bulk concentration of substrates
    model.dS_0dt = dae.DerivativeVar(model.S_0, wrt=model.time) # first-derivative of S_0 on time
    
    # Pore-scale variables (for BVP) index order:-> Components, x, time
    model.S_n = pyo.Var(model.Components, model.x, model.time)      # S_n, Pore concentration Substrate 1 in pore
    model.dS_ndx   = dae.DerivativeVar(model.S_n, wrt=model.x)      # first-derivative of S_n
    model.d2S_ndx2 = dae.DerivativeVar(model.dS_ndx, wrt=model.x)   # second-derivative of S_n
    
    # Add constraints and objectives
    add_bvp_constraints(model, sid=sid, decay=decay)
    add_reactor_odes(model)
    
    return model

if __name__ == "__main__":
    try:

        print("Building test model...")
        test_model = build_reactor_model(sid='delta', decay=False)
        
        print("Solving test model...")
        solved_model, solver_results = solve_model(test_model)
        
        # Print some basic results
        if solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("✓ Optimization successful!")
            # Result print
            print("Final S2:", test_model.S_0['S2',test_model.time.last()]()/test_model.S_0['S1',test_model.time.first()]())
            print("Final S3:", test_model.S_0['S3',test_model.time.last()]()/test_model.S_0['S1',test_model.time.first()]())
            # Generate individual plot
            print("\n1. Plotting enzyme profiles...")
            viz.plot_enzyme_pore_profiles(solved_model)
            
            print("\n2. Plotting substrate concentrations...")
            viz.plot_substrate_time_profiles(solved_model)
        else:
            print("✗ Optimization failed or did not converge optimally")
            print(f"Termination condition: {solver_results.solver.termination_condition}")
            
    except ImportError as e:
        print(f"Could not import model builder: {e}")
        print("This is normal if model.py is not in the same directory.")
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Test completed.")
