import pyomo.environ as pyo
import pyomo.dae as dae

def enzyme_profile_rule(model, E_max, start=1, end=0, fun='linear', **kwargs):
    """
    Add configurable enzyme distribution
    
    Parameters:
    fun: enzyme density distribution function throughout pore length (density between 0 and maximum enzyme density m.EA & m.EB)
            - linear    : linear function
            - exp       : exponential function (must define exp_const)
            - step      : step function (must define x_step as 0 to 1)
            
    start: Density at x=0 as fraction of maximum enzyme density (0 to 1)
    end: Density at x=L as fraction of maximum enzyme density (0 to 1)
    """


    if fun == 'linear':
        def profile_rule(m, x):
            return E_max * (start + (end - start) * (x/m.L))
        return pyo.Expression(model.x, rule=profile_rule)
    else:
        raise ValueError(f"Unsupported profile type: {fun}")
    # if fun == 'linear':
    #     """
    #     General linear enzyme profile.
    #     E(x) = E0 * [start + (end - start) * (x / L)]
    #     """
    #     return E_max * (start + (end - start) * (x/L))
    
    # elif fun == 'exp':
    #     pass
    # elif fun == 'step':
    #     pass
    # else:
    #     raise Exception("Error: function type is not implemented yet.")
    

def solve_model(model):
    """
    Discretize and solve the reactor kinetics model.
    
    Parameters:
    model: Pyomo model to solve
    
    Returns:
    model: Solved model
    results: Solver results
    """
    print("Discretizing model...")
    
    # Discretize time and space (x) variables
    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(model, wrt=model.time, nfe=40, ncp=2)  # Total discretization points = nfe*ncp
    discretizer.apply_to(model, wrt=model.x, nfe=20, ncp=3)
    
    print("Discretization completed.")
    print("Solving model with IPOPT...")
    
    # Solve model
    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(model, tee=True)
    
    print(f"Solver termination condition: {results.solver.termination_condition}")
    print(f"Solver status: {results.solver.status}")
    
    return model, results