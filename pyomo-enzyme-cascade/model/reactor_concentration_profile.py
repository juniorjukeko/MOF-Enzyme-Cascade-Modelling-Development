import pyomo.environ as pyo
import pyomo.dae as dae

def add_reactor_odes(model, immobilization='co-immobilization'):
    """
    Add reactor-scale ODEs for different immobilization schemes
    """
    
    if immobilization == 'co-immobilization':
        # print("Debug: Setting up CO-IMMOBILIZATION reactor ODEs...")
        # Co-immobilization: both enzymes in same pores
        # S1 -> S2 -> S3 in the same pores
        
        def S1_reactor_ivp_rule(m, t):
            return m.dS_0dt['S1', t] == -m.flux['S1', t]  # S1 consumed
        
        def S2_reactor_ivp_rule(m, t):
            # S2: produced from S1, consumed to S3 (same pores)
            return m.dS_0dt['S2', t] == m.flux['S1', t] - m.flux['S2', t]
        
        def S3_reactor_ivp_rule(m, t):
            return m.dS_0dt['S3', t] == m.flux['S2', t]  # S3 produced from S2
        
        model.S1_reactor_ivp = pyo.Constraint(model.time, rule=S1_reactor_ivp_rule)
        model.S2_reactor_ivp = pyo.Constraint(model.time, rule=S2_reactor_ivp_rule)
        model.S3_reactor_ivp = pyo.Constraint(model.time, rule=S3_reactor_ivp_rule)
        
    elif immobilization == 'single':
        # print("Debug: Setting up SINGLE reactor ODEs...")
        # Single immobilization: enzymes in separate pores
        # Type A pores: S1 -> S2
        # Type B pores: S2 -> S3
        
        def S1_reactor_ivp_rule(m, t):
            # S1 only consumed in Type A pores
            return m.dS_0dt['S1', t] == -m.flux['S1', t]
        
        def S2_reactor_ivp_rule(m, t):
            # S2: produced in Type A pores, consumed in Type B pores
            return m.dS_0dt['S2', t] == m.flux['S1', t] - m.flux['S2', t]
        
        def S3_reactor_ivp_rule(m, t):
            # S3 only produced in Type B pores  
            return m.dS_0dt['S3', t] == m.flux['S2', t]
        
        model.S1_reactor_ivp = pyo.Constraint(model.time, rule=S1_reactor_ivp_rule)
        model.S2_reactor_ivp = pyo.Constraint(model.time, rule=S2_reactor_ivp_rule)
        model.S3_reactor_ivp = pyo.Constraint(model.time, rule=S3_reactor_ivp_rule)
    
    else:
        raise ValueError(f"Invalid immobilization scheme: {immobilization}")
    
    # --- Initial conditions ---
    def ic_S_0_rule(m, component):
        return m.S_0[component, 0] == m.S_initial[component]
    model.ic_S_0 = pyo.Constraint(model.Components, rule=ic_S_0_rule)
