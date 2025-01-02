import numpy as np

static_config = {
    "ws": "fert_optimization",
    "desc": "Fertilization optimization",

    # The variables that define the initial state of the model
    "initial_wc": 0.1,
    "initial_conc": 0,
    "croot_max": 40,
    "top": 0,  # depth of surface
    "bottom": -100,  # depth of bottom in METERS
    "dx": 1,
    "hydro_pressure": 0.12,
    "conc": 0,
    "root_distribution": lambda root_depth: np.linspace(1, 0, int(root_depth)),  # root distribution decrease linearly from 1 to 0 in the root depth
    "root_distribution_fill": lambda root_distribution, profile: np.concatenate((root_distribution, np.zeros(len(profile) - len(root_distribution)))),  # root distribution is 0 from the root_depth to the end
    # Atm pressure variables
    "fertigation_conc": 40,  # N-NO3
    "daily_et": 0.2,  # output flux of water by evaporation or by plant uptake 
    "n_days": 30,  # choose between 1 to 31
    "n_hours": 24 * 30,
    "atm_columns": ["tAtm", "Prec", "rSoil", "rRoot", "hCritA", "rB", "hB", "ht", "tTop", "tBot", "Ampl", "cTop", "cBot"],
    "PREC": 1,
    "CTOP": 11,
    "irrigation_func": lambda daily_et, leaching_fraction: daily_et * leaching_fraction,
    "precipitation_interval": lambda n_days: [6 + (i * 24) for i in range(n_days)],  # hours in which precipitation occurs
    "transpiration_frac": 0.9,
    "evaporation_frac": 0.1,
    
    # Material information
    "l": -0.5,
    "nitrate_trans": (1.5, 10, 1, 0), 

    # Solute information
    "sol_beta": 1,
    "sol_difw": 0.068,

    # Waterflow information
    "VAN_GENUCH_6_PARAM": 0,
    "ATM_W_SURFACE_RUNOFF": 3,
    "FREE_DRAINAGE": 4,

    # Solute transport information
    "EQ_SOLUTE_TRANPORT": 0,
    "CAUCHY_BOUNDRY_COND": -1,
    "CONT_CONC_PROFILE": 0,

    # Root uptake information
    "FEDES_ET_AL": 0,
    "p0": -10,
    "p2h": -400,
    "p2l": -600,
    "p3": -8000,
    "r2h": 0.02,
    "r2l": 0.004,
    "poptm": [-25],

    # Observation node
    "DEPTHS": [10]
}