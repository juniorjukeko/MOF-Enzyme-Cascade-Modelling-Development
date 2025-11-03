import pyomo.environ as pyo
import pyomo.dae as dae

def enzyme_profile_rule(model, E_max, start=1, end=0, fun='linear', **kwargs):
    """
    Add configurable enzyme distribution
    
    Parameters:
    fun: enzyme density distribution function throughout pore length (density between 0 and maximum enzyme density m.EA & m.EB)
            - linear    : linear function
            - step      : step function (must define x_step as 0 to 1)
    start: 
        - linear    : Density at x=0 as fraction of maximum enzyme density (0 to 1)
        - step      : Baseline enzyme density (before step-up and after step-down) as fraction of maximum enzyme density (0 to 1) 
    end: 
        - linear    : Density at x=L as fraction of maximum enzyme density (0 to 1)
        - step      : Plateau enzyme density (after step-up and before step-down) as fraction of maximum enzyme density (0 to 1)
    kwargs: Additional parameters for specific distributions
        - For 'step': 
            x_step_up: Fraction of L where step up begins (0 to 1)
            x_step_down: Fraction of L where step down begins (0 to 1)
            smoothness: Smoothness factor for transitions (lower = smoother, default=100)
    """
    # Validate start and end parameters
    if (start < 0 or start > 1) or (end < 0 or end > 1):
        raise ValueError("'start' and 'end' must be between 0 and 1")

    if fun == 'linear':
        def profile_rule(m, x):
            return E_max * (start + (end - start) * (x/m.L))
        return pyo.Expression(model.x, rule=profile_rule)
    
    elif fun == 'step':
        x_step_up   = kwargs.get('x_step_up', 0.3)      # Start of step-up transition
        x_step_down = kwargs.get('x_step_down', 0.7)    # Start of step-down transition
        smoothness  = kwargs.get('smoothness', 100.0)   # Smoothness factor
        
        # Validate step parameters
        if x_step_up < 0 or x_step_up > 1:
            raise ValueError("x_step_up must be between 0 and 1")
        if x_step_down < 0 or x_step_down > 1:
            raise ValueError("x_step_down must be between 0 and 1")
        if x_step_up >= x_step_down:
            raise ValueError("x_step_up must be less than x_step_down")

        
        def profile_rule(m, x):
            x_frac = x / m.L

            # Smooth step-up transition (sigmoid function)
            step_up_transition   = 1.0 / (1.0 + pyo.exp(-smoothness * (x_frac - x_step_up)))  
            
            # Smooth step-down transition (sigmoid function)
            step_down_transition = 1.0 / (1.0 + pyo.exp(-smoothness * (x_frac - x_step_down)))
            
            # Combined profile:
            # - Before step_up: Constant at start value (before step-up) -> start * E_max
            # - During step_up transition: Smooth step-up transition to (end)
            # - Plateau: Constant at step_value (plateau) -> end * E_max  
            # - During step_down transition: Smooth step-down transition to (start)
            # - After step_down: Constant at start value -> start * E_max

            # (before step_up + (transition->plateau->transition) + after step_down)
            step_profile = (start * E_max * (1.0 - step_up_transition) +
                            end * E_max * (step_up_transition - step_down_transition) +
                            start * E_max * step_down_transition)
            return step_profile
        return pyo.Expression(model.x, rule=profile_rule)
                

    else:
        raise ValueError(f"Unsupported profile type: {fun}")

def calculate_pore_count_coefficient(model, E_profile_expression, E_max):
    """
    Calculate pore count coefficient based on area under enzyme profile curve.
    
    The coefficient normalizes the pore count so that it's equivalent to having 
    constant E_max from x=0 to x=L (alpha SID).
    
    Parameters:
    model: Pyomo model object
    E_profile_expression: Pyomo Expression for enzyme profile (E_x_profile)
    E_max: Maximum enzyme loading parameter
    
    Returns:
    pore_count_coef: Coefficient to multiply Np by in bvp flux rule
    """
    x_values = sorted(list(model.x))
    
    # Sum of enzyme concentrations at discretization points
    total_enzyme = 0.0
    count = 0
    
    for x in x_values:
        try:
            # Use Pyomo's value() function to safely get numerical values
            enzyme_val = pyo.value(E_profile_expression[x])
            total_enzyme += enzyme_val
            count += 1
        except:
            continue
    
    # Average enzyme concentration
    avg_enzyme = total_enzyme / count if count > 0 else 0
    
    # Reference average (constant E_max) - use value() for Pyomo parameters
    ref_enzyme_avg = pyo.value(E_max)
    
    # Coefficient is ratio of averages
    pore_count_coef = ref_enzyme_avg / avg_enzyme if avg_enzyme > 0 else 1.0
    
    print(f"Pore count coefficient (simple):")
    print(f"  Average enzyme: {avg_enzyme:.6e}")
    print(f"  Reference average: {ref_enzyme_avg:.6e}")
    print(f"  Pore count coefficient: {pore_count_coef:.6f}")
    
    return pore_count_coef
 