# model.py
import pyomo.environ as pyo
import pyomo.dae as dae

import params_initialization
from model.pore_concentration_profile import add_bvp_constraints
from model.reactor_concentration_profile import add_reactor_odes

def build_reactor_model(immobilization='co-immobilization', decay_coef={'kA':0, 'kB':0}, **kwargs):
    model = pyo.ConcreteModel()
    
    # Model parameters indexing - one stage, 3 substrates
    model.Stage      = pyo.Set(initialize=[1])  # Single stage
    model.Components = pyo.Set(initialize=['S1', 'S2', 'S3']) # Substrate components

    # Load parameters
    model = params_initialization.load_parameters(model)  # load_parameters.py

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

    bvp_kwargs = kwargs.get('bvp_kwargs')
    # Add constraints and objectives
    # print(immobilization)
    add_bvp_constraints(model, immobilization=immobilization, decay_coef=decay_coef, bvp_kwargs=bvp_kwargs)
    add_reactor_odes(model, immobilization=immobilization)
    
    return model

if __name__ == "__main__":
    from model.solve import solve_model_robust
    import visualization.model_visualization as m_viz
    try:
        print("Building test model...")
        bvp_kwargs = {
            'default_fun': 'linear', 
            'adjust_Np': False,
            'enzymeA': {
                'fun': 'linear',
                'start': 1, 
                'end': 0, 
                'x_step_up': 0.0, 
                'x_step_down': 0.5, 
                'smoothness': 50
            },
            'enzymeB': {
                'fun': 'linear',
                'start': 0, 
                'end': 1,
                'x_step_up': 0.6, 
                'x_step_down': 1.0,
                'smoothness': 80
            }
        }

        # decay_coef = {'kA': 0.004, 'kB': 0.002} # Enzyme A, Enzyme B decay kinetics coefficient
        decay_coef = {'kA': 0.001, 'kB': 0.001}
        test_model = build_reactor_model(immobilization='single', 
                                         decay_coef=decay_coef, bvp_kwargs=bvp_kwargs)
        
        # Future implementation : If needed to do initial states 
        # initialize_model_states(model, immobilization=immobilization)
        print("Solving test model...")
        solved_model, solver_results = solve_model_robust(test_model, max_iter=1000, tol=1e-4, verbose=True)
        
        # Print some basic results
        if solver_results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print("[DONE] Optimization successful!")
            # Result print
            print("Final S2 yield:", test_model.S_0['S2',test_model.time.last()]()/test_model.S_0['S1',test_model.time.first()]())
            print("Final S3 yield:", test_model.S_0['S3',test_model.time.last()]()/test_model.S_0['S1',test_model.time.first()]())
            
            # Generate individual plot
            # import matplotlib.pyplot as plt
            print("\n1. Plotting enzyme profiles...") 
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8)) 
            m_viz.plot_enzyme_pore_profiles(solved_model,immobilization='single') 
            m_viz.plot_enzyme_decay_profiles(solved_model, decay_coef) 
            print("\n2. Plotting substrate concentrations...") 
            m_viz.plot_substrate_time_profiles(solved_model) 
            
        else:
            print("[FAIL] Optimization failed or did not converge optimally")
            print(f"Termination condition: {solver_results.solver.termination_condition}")
            
    except ImportError as e:
        print(f"Could not import model builder: {e}")
        print("This is normal if model.py is not in the same directory.")
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("Test completed.")
