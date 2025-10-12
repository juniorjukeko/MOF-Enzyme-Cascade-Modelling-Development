import pyomo.environ as pyo
import pyomo.dae as dae

# based on ivp.py
def add_reactor_odes(model):
    # --- ODE system for material balances (Equation 2) ---
    def S1_reactor_ivp_rule(m, t):
        return m.dS_0dt['S1', t] == -m.flux['S1', t]
    
    model.S1_reactor_ivp = pyo.Constraint(model.time, rule=S1_reactor_ivp_rule)
    
    def S2_reactor_ivp_rule(m, t):
        return m.dS_0dt['S2', t] == m.flux['S1', t] - m.flux['S2', t]
    
    model.S2_reactor_ivp = pyo.Constraint(model.time, rule=S2_reactor_ivp_rule)
    
    def S3_reactor_ivp_rule(m, t):
        return m.dS_0dt['S3', t] == m.flux['S2', t] 
    
    model.S3_reactor_ivp = pyo.Constraint(model.time, rule=S3_reactor_ivp_rule)
    
    # --- Initial conditions (config.py)---
    # Condition #1: Initial substrate bulk conc.
    def ic_S_0_rule(m, component):
        return m.S_0[component, 0] == m.S_initial[component]
    model.ic_S_0 = pyo.Constraint(model.Components, rule=ic_S_0_rule)


