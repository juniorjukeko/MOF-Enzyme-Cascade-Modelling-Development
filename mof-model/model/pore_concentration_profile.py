# NOTE: Modify the bvp_rule with E_A/B_max function (x-dependent)
# Based on bvp.py
import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
from .utils import enzyme_profile_rule

def enzyme_decay_kinetics():
    # NOTE: add decay params in the future
    pass

def add_bvp_constraints(model, sid='alpha', decay=False): 
    # --- ODE system (Equations 3, 6, 7) ---
    # Diffusion-reaction ODE in pore alpha-A (Enzyme A only)
    if sid == 'alpha':
        def typeA_pore_bvp_rule(m, x, t):
            return m.D['S1'] * m.d2S_ndx2['S1', x, t] == (
                m.EA * m.kA * m.S_n['S1', x, t]
            )   
        model.typeA_pore_bvp = pyo.Constraint(model.x, model.time, rule=typeA_pore_bvp_rule)
        
        # Diffusion-reaction ODE in pore alpha-B (Enzyme B only)
        def typeB_pore_bvp_rule(m, x, t):
            return m.D['S2'] * m.d2S_ndx2['S2', x, t] == (
                m.EB * m.kB * m.S_n['S2', x, t]
            )
        model.typeB_pore_bvp = pyo.Constraint(model.x, model.time, rule=typeB_pore_bvp_rule)
    
    elif sid in ['gamma', 'delta']:
        if sid == 'gamma':
            model.EA_x_profile = enzyme_profile_rule(model, model.EA, start=1, end=0, fun='linear')
            model.EB_x_profile = enzyme_profile_rule(model, model.EB, start=0, end=1, fun='linear')
        elif sid == 'delta':
            model.EA_x_profile = enzyme_profile_rule(model, model.EA, start=0, end=1, fun='linear')
            model.EB_x_profile = enzyme_profile_rule(model, model.EB, start=1, end=0, fun='linear')
            
        def S1_mixed_pore_bvp_rule(m, x, t):
            return m.D['S1'] * m.d2S_ndx2['S1', x, t] == (
                m.EA_x_profile[x] * m.kA * m.S_n['S1', x, t]
            )
        model.S1_mixed_pore_bvp = pyo.Constraint(model.x, model.time, rule=S1_mixed_pore_bvp_rule)  
              
        def S2_mixed_pore_bvp_rule(m, x, t):
            return m.D['S2'] * m.d2S_ndx2['S2', x, t] == (
                m.EB_x_profile[x] * m.kA * m.S_n['S2', x, t] - m.EA_x_profile[x] * m.kA * m.S_n['S1', x, t]
            )
        model.S2_mixed_pore_bvp = pyo.Constraint(model.x, model.time, rule=S2_mixed_pore_bvp_rule)
        
        def S3_mixed_pore_bvp_rule(m, x, t):
            return m.D['S3'] * m.d2S_ndx2['S3', x, t] == (
                m.EB_x_profile[x] * m.kA * m.S_n['S2', x, t]
            )
        model.S3_mixed_pore_bvp = pyo.Constraint(model.x, model.time, rule=S3_mixed_pore_bvp_rule)         

    # --- Boundary conditions (Equations 4, 5) ---
    # Rule #1: substrate conc. at x=0 equal to bulk conc.
    def bc1_rule(m, component, t):
        return m.S_n[component, 0, t] == m.S_0[component,t]
    model.bc1 = pyo.Constraint(model.Components, model.time, rule=bc1_rule)
    
    # Rule #2: conc. gradient at x=L is zero (no diffusion flux)
    def bc2_rule(m, component, t):
        return m.dS_ndx[component, m.L, t] == 0
    model.bc2 = pyo.Constraint(model.Components, model.time, rule=bc2_rule)
    
    # Flux expression (used in reactor ODE) --> Equation 2 Right-hand side
    # Pore ratio TypeA : TypeB = 50:50
    #     vI = -A * D1 * Y11[0] * Np
    # vII = A * D3 * Y31[0] * Np 
    def flux_rule(m, component, t):
        return -m.D[component] * m.dS_ndx[component, 0, t] * m.A * m.Np
    model.flux = pyo.Expression(model.Components, model.time, rule=flux_rule)

