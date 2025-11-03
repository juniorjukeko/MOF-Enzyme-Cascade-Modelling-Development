# pore_concentration_profile.py
import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
from .utils import enzyme_profile_rule, calculate_pore_count_coefficient

def add_bvp_constraints(model, immobilization='co-immobilization', decay_coef={'kA':0, 'kB':0}, bvp_kwargs=None): 
    # --- Create symbolic enzyme decay profiles (Expressions) ---
    kA_decay = decay_coef.get('kA', 0.0)
    kB_decay = decay_coef.get('kB', 0.0)

    if kA_decay > 0:
        model.decay_A = pyo.Expression(model.time, rule=lambda m, t: pyo.exp(-kA_decay * t))
    else:
        model.decay_A = pyo.Expression(model.time, rule=lambda m, t: 1.0)

    if kB_decay > 0:
        model.decay_B = pyo.Expression(model.time, rule=lambda m, t: pyo.exp(-kB_decay * t))
    else:
        model.decay_B = pyo.Expression(model.time, rule=lambda m, t: 1.0)
        
    if immobilization == 'single':
        
        # --- ODE system (Equations 3, 6, 7) ---
        # Diffusion-reaction ODE in pore alpha-A (Enzyme A only)
        def typeA_pore_bvp_rule(m, x, t):
            return m.D['S1'] * m.d2S_ndx2['S1', x, t] == (
                m.EA * m.decay_A[t] * m.kA * m.S_n['S1', x, t]
            )   
        model.typeA_pore_bvp = pyo.Constraint(model.x, model.time, rule=typeA_pore_bvp_rule)
        
        # Diffusion-reaction ODE in pore alpha-B (Enzyme B only)
        def typeB_pore_bvp_rule(m, x, t):
            return m.D['S2'] * m.d2S_ndx2['S2', x, t] == (
                m.EB * m.decay_B[t] * m.kB * m.S_n['S2', x, t]
            )
        model.typeB_pore_bvp = pyo.Constraint(model.x, model.time, rule=typeB_pore_bvp_rule)
    
    elif immobilization == 'co-immobilization':
        # Get specific enzyme kwargs for enzyme density profile
        default_fun    = bvp_kwargs.get('default_fun', 'linear')
        enzymeA_kwargs = bvp_kwargs.get('enzymeA', {})
        enzymeB_kwargs = bvp_kwargs.get('enzymeB', {})

        EA_fun = enzymeA_kwargs.get('fun', default_fun)
        EB_fun = enzymeB_kwargs.get('fun', default_fun) 
        
        # Enzyme A profile: High at entry, low at end of pores (default). 
        model.EA_x_profile = enzyme_profile_rule(
            model,
            model.EA,
            start=enzymeA_kwargs.get('start', 1),
            end=enzymeA_kwargs.get('end', 0),
            fun=EA_fun,
            **{k: v for k, v in enzymeA_kwargs.items() if k not in ['fun', 'start', 'end']}
        )
        # Enzyme B: low at entry, high at end of pores (default). 
        model.EB_x_profile = enzyme_profile_rule(
            model,
            model.EB,
            start=enzymeB_kwargs.get('start', 0),
            end=enzymeB_kwargs.get('end', 1),
            fun=EB_fun,
            **{k: v for k, v in enzymeB_kwargs.items() if k not in ['fun', 'start', 'end']}
        )
        # --- ODE system (Equations 3, 6, 7) --- 
        def S1_mixed_pore_bvp_rule(m, x, t):
            return m.D['S1'] * m.d2S_ndx2['S1', x, t] == (
                m.EA_x_profile[x] * m.decay_A[t] * m.kA * m.S_n['S1', x, t]
            )
        model.S1_mixed_pore_bvp = pyo.Constraint(model.x, model.time, rule=S1_mixed_pore_bvp_rule)  
              
        def S2_mixed_pore_bvp_rule(m, x, t):
            return m.D['S2'] * m.d2S_ndx2['S2', x, t] == (
                m.EB_x_profile[x] * m.decay_B[t] * m.kB * m.S_n['S2', x, t] 
                - m.EA_x_profile[x] * m.decay_A[t] * m.kA * m.S_n['S1', x, t]
            )
        model.S2_mixed_pore_bvp = pyo.Constraint(model.x, model.time, rule=S2_mixed_pore_bvp_rule)
        
        def S3_mixed_pore_bvp_rule(m, x, t):
            return m.D['S3'] * m.d2S_ndx2['S3', x, t] == (
                m.EB_x_profile[x] * m.decay_B[t] * m.kB * m.S_n['S2', x, t]
            )
        model.S3_mixed_pore_bvp = pyo.Constraint(model.x, model.time, rule=S3_mixed_pore_bvp_rule)         
    else:
        raise Exception("Invalid immobilization scheme!")
    
    # --- Boundary conditions (Equations 4, 5) ---
    # Rule #1: substrate conc. at x=0 equal to bulk conc.
    def bc1_rule(m, component, t):
        return m.S_n[component, m.x.first(), t] == m.S_0[component,t]
    model.bc1 = pyo.Constraint(model.Components, model.time, rule=bc1_rule)
    
    # Rule #2: conc. gradient at x=L is zero (no diffusion flux)
    def bc2_rule(m, component, t):
        return m.dS_ndx[component, m.x.last(), t] == 0
    model.bc2 = pyo.Constraint(model.Components, model.time, rule=bc2_rule)
    
    # Flux expression (used in reactor ODE) --> Equation 2 Right-hand side    
    if immobilization == 'single':
        # Pore ratio TypeA : TypeB = 50:50
        def flux_rule(m, component, t):
            return -m.D[component] * m.dS_ndx[component, m.x.first(), t] * m.A * m.Np / 2
 
        model.flux = pyo.Expression(model.Components, model.time, rule=flux_rule)
        
    elif immobilization == 'co-immobilization':
        # Whether to adjust pore number to get equal total activity
        if bvp_kwargs.get('adjust_Np', False):
            pore_count_coef = calculate_pore_count_coefficient(model, model.EA_x_profile, model.EA)
        else:
            pore_count_coef = 2
            
        def flux_rule(m, component, t):
            return -m.D[component] * m.dS_ndx[component, 0, t] * m.A * m.Np * pore_count_coef / 2

        model.flux = pyo.Expression(model.Components, model.time, rule=flux_rule)
