import pyomo.environ as pyo
import config
model = pyo.ConcreteModel()
# Parameters initialization for Alpha-SID (Simultaneous)
def load_parameters_alpha(model):
   
    # --- Batch reactor parameters initialization ---
    # a. Initial concentrations (single stage)
    S_initial = {}
    S_initial['S1'], S_initial['S2'], S_initial['S3'] = config.INITIAL_SUBSTRATE_CONC.values()
    model.S_initial = pyo.Param(model.Components, initialize=S_initial)
    
    # b. Operation parameters - single time frame
    model.tf = pyo.Param(initialize=config.REACTION_TIME)  # Total reaction time (minutes)
    model.Np = pyo.Param(initialize=config.PORES_COUNT)    # Number of pores
    
    # --- Geometric parameters initialization ---
    # a. Pore geometry
    model.L  = pyo.Param(initialize=config.PORE_LENGTH)    # Pore length (dm)
    model.A  = pyo.Param(initialize=config.PORE_AREA)      # Cross-sectional area (dm²)
    
    # b. Enzyme loadings - both active simultaneously
    model.EA = pyo.Param(initialize=config.MAX_ENZYME_SURFACE_DENSITY['Enzyme_A'])  # (μmol/dm²)
    model.EB = pyo.Param(initialize=config.MAX_ENZYME_SURFACE_DENSITY['Enzyme_B'])  # (μmol/dm²)
     
    # --- Kinetic parameters - both enzymes active simultaneously ---
    model.kA = pyo.Param(initialize=config.REACTION_CONST['Enzyme_A'])   # Enzyme A first-order rate, or V_max for MM (dm²/μmol/min)
    model.kB = pyo.Param(initialize=config.REACTION_CONST['Enzyme_B'])   # Enzyme B first-order rate, or V_max for MM (dm²/μmol/min)
    
    model.kM_A = pyo.Param(initialize=config.MICHAELIS_MENTEN_CONST['Enzyme_A'])    # Enzyme A MM constant (μM)
    model.kM_B = pyo.Param(initialize=config.MICHAELIS_MENTEN_CONST['Enzyme_B'])    # Enzyme B MM constant (μM)
    
    # --- Physical property parameters initialization ---
    D_config = config.DIFFUSIFITY_CONST
    model.D = pyo.Param(model.Components, initialize=D_config)  # (dm²/min)
       
    return model