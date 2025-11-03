"""
pore_concentration_profile.py

Simulates the concentration profiles of substrates inside a SID-alpha pore
by solving boundary value problems (BVPs) from equations (3) to (7) in the paper
"""

import numpy as np
from scipy.integrate import solve_bvp
import scipy.interpolate as inter

def simulate_pore_concentration(
    batch_reactor_params: dict,
    physical_prop_params: dict,
    pore_geometric_params: dict,
    reaction_kinetic_params: dict,
    counter: int,
    init_guess: dict,
):
    """
    Solve the concentration profile of substrate along cylindrical pore length inside the pore.

    Parameters
    ----------
    batch_reactor_params : dict
        {
            "substrates": {"S1": float, "S2": float, "S3": float},
            "tf": float,
            "Np": int
        }
    physical_prop_params : dict
        {
            "D": {"S1": float, "S2": float, "S3": float},
        }
    pore_geometric_params : dict
        {
            "A": float, <-- reference: config.PORE_AREA
            "L": float, <-- reference: config.PORE_LENGTH
            "E_max": {"E_max_A": float, "E_max_B": float} <-- reference: config.MAX_ENZYME_SURFACE_DENSITY
        }

    reaction_kinetic_params : dict
        {
            "kA": float,
            "kB": float
        }

    D1, D2, D3 : float
        Diffusion coefficients for substrates 1–3.
    kM : float
        Michaelis–Menten constant for enzyme A.
    counter : int
        Iteration counter for updating guesses.
    init_guess : dict
        Initial guess dictionary:
        {
            "S1": {0: [...values...], 1: [...derivatives...]},
            "S2": {...},
            "S3": {...}
        }

    Returns
    -------
    vI : float
        Substrate flux at pore entrance.
    solution : Bunch
        Scipy solve_bvp solution object.
    new_guess : dict
        Updated initial guess dictionary for next iteration.
    """

    x = np.linspace(0, L, 50)

    def ode_system(x, Y):
        """
        Differential equations for substrate S1.
        Y[0] = S1 profile
        Y[1] = dS1/dx
        """
        dS1dx = Y[1]
        dS1dx2 = (EA * Y[0] * kA) / (D1 * (kM + Y[0]))
        return np.vstack([dS1dx, dS1dx2])

    def boundary_conditions(Y0, YL):
        """
        Boundary conditions:
        - At x=0: concentration equals bulk S1o
        - At x=L: derivative (flux) goes to zero
        """
        return np.array([Y0[0] - S1_init, YL[1]])

    # Prepare initial guess
    y_guess = np.vstack([init_guess["S1"][0], init_guess["S1"][1]])

    # Solve the BVP
    sol = solve_bvp(
        ode_system,
        boundary_conditions,
        x,
        y_guess,
        max_nodes=1000,
    )

    Y1, dY1dx = sol.y

    # Update guesses for next iteration (if needed)
    new_guess = init_guess.copy()
    if counter < 10 and sol.status == 0:
        f0 = inter.interp1d(sol.x, sol.y[0, :], kind="linear")
        f1 = inter.interp1d(sol.x, sol.y[1, :], kind="linear")
        new_guess["S1"][0] = f0(x)
        new_guess["S1"][1] = f1(x)

    # Calculate substrate flux at pore entrance
    vI = -A * D1 * dY1dx[0] * Np / 2

    return vI, sol, new_guess