import pyomo.environ as pyo
import pyomo.dae as dae

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