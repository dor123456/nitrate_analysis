import numpy as np

static_config = {
    "ws": "fert_optimization",
    "desc": "Fertilization optimization",

    # The variables that define the initial state of the model
    "initial_wc_10": 0.185,
    "initial_wc_40": 0.3,
    "auto_wc_and_NO3" : True,
    "initial_wc_distribution": lambda resid_wc, wc_10, wc_40, sat_wc, profile: np.concatenate((np.linspace(3*resid_wc, wc_10, 10), np.linspace(wc_10, wc_40, 30), np.linspace(wc_40, 0.9*sat_wc, len(profile)-40))),
    "initial_conc": 0,
    "initial_conc_distribution": lambda initial_conc, profile: np.full(len(profile), initial_conc), # finish this for starting stage 
    "croot_max": 40,
    "top": 0,  # depth of surface
    "bottom": -49,  # depth of bottom in cm
    "dx": 1,
    "root_distribution": lambda root_depth: np.linspace(1, 0, int(root_depth)),  # root distribution decrease linearly from 1 to 0 in the root depth
    "root_distribution_fill": lambda root_distribution, profile: np.concatenate((root_distribution, np.zeros(len(profile) - len(root_distribution)))),  # root distribution is 0 from the root_depth to the end
    # Atm pressure variables
    "daily_et": 0.2,  # output flux of water by evaporation or by plant uptake 
    "n_days": 12,  # choose between 1 to 31
    "n_hours": 24 * 12,
    "atm_columns": ["tAtm", "Prec", "rSoil", "rRoot", "hCritA", "rB", "hB", "ht", "tTop", "tBot", "Ampl", "cTop", "cBot"],
    "PREC": 1,
    "CTOP": 11,
    "irrigation_func": lambda daily_et, leaching_fraction: daily_et * leaching_fraction,
    "precipitation_interval": lambda n_days: [6 + (i * 24) for i in range(n_days)],  # hours in which precipitation occurs
    "transpiration_frac": 0.001,
    "evaporation_frac": 0.999,
    
    # Material information
    "l": -0.5,
    "nitrate_trans": (1.5, 10, 1, 0), # bulk.d +-0.3 DisperL 10, frac, mobile_wc

    # Solute information
    "sol_beta": 1,
    "sol_difw": 0.068,

    # Waterflow information
    "VAN_GENUCH_6_PARAM": 0,
    "ATM_W_SURFACE_LAYER": 2,
    "SEEPAGE_FACE": 6,

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
    "DEPTHS": [-10, -20, -40]
}