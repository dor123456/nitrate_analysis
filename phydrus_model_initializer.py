import os
import pandas as pd
import numpy as np
import phydrus as ps
import matplotlib.pyplot as plt
import time
from collections import UserDict

class DynamicConfig(UserDict):
    # Example usages
    default_config = {
        # soil variables
        "h_conductivity" : 1,
        "resid_wc" : 0.075,
        "sat_wc" : 0.3,
        "alpha" : 0.1075,
        "n_empiric" : 2.285,
        # plant variables
        "root_depth" : 20,
        "leaching_fraction" : 1,
        #precipitaion
        "precipitation" : 10
    }
    
    def __init__(self, defaults=default_config):
        super().__init__(defaults)

class PhydrusModelInit():
    """
    This class initializes and runs a phydrus model. 
    The default initizlizes a model considering, water, one solute, and plants.
    The class gets a static configuration dict representing , the initizlixeing functions and the main code
    all lengths are in cm
    """
    static_config = {}
    dynamic_config = None
    ml = None
    def __init__(self, dynamic_config, static_config):
        # input units: cm, ppm, please note to unit convertion in the output of the script
        self.dynamic_config = dynamic_config # changing variables
        self.static_config = static_config # name of csv file containing precipitation info
        self.create_model()
        self.initialize_model()
        self.run_model

    def create_model(self): #
        exe = os.path.join(os.getcwd(), "../hydrus.exe") #this never change!!
        # Create the basic model
        self.ml = ps.Model(exe_name=exe, ws_name=self.static_config["ws"], name="model", description=self.static_config["desc"],
                    mass_units="M", time_unit="hours", length_unit="mm")

    def calculate_applied_N(self):
        return self.static_config["daily_et"] * self.dynamic_config["leaching_fraction"] * self.static_config["fertigation_conc"] * self.dynamic_config["n_days"]

    def add_fake_precipitation(self, atm):
        irrigation = self.static_config["irrigation_func"](self.static_config["daily_et"], self.static_config["leaching_fraction"])
        atm.iloc[[self.static_config["precipitation_interval"], self.static_config["PREC"]]] = irrigation
        atm.iloc[[self.static_config["precipitation_interval"], self.static_config["CTOP"]]] = self.static_config["fertigation_conc"] #ctop

    def add_real_precipitaion(self, atm):
        return

    def linear_distribute_ET(self, daily_ET, days=1):
        ET = np.zeros(24)
        ET[7:18] = daily_ET/(18-7)
        transpiration = self.static_config["transpiration_frac"] * ET
        evaporation =self.static_config["evaporation_frac"] * ET
        hours = np.arange(24*days)
        # Create the DataFrame with repeated values
        return pd.DataFrame({
            'hour': hours,
            'evapotranspiration': np.tile(ET, n),
            'transpiration': np.tile(transpiration, n),
            'evaporation': np.tile(evaporation, n)
        })
    
    def add_atm_pressure(self):
        # (module) -> None
        # =============================================================================
        # atm_bc
        # =============================================================================
        ml = self.ml
        ET = self.linear_distribute_ET(self.static_config["daily_et"], self.static_config["n_days"])
        atm = pd.DataFrame(0, index=np.arange(self.static_config["n_hours"]), columns=self.static_config["atm_columns"]) # add columns according to the defined n_days

        atm['tAtm'] = np.arange(1,self.static_config["n_hours"]+1)
        atm['rSoil'] = ET['evaporation']
        atm['rRoot'] = ET['transpiration']
        atm['hCritA'] = 1000000 # random big number (???)

        if self.dynamic_config["precipitaion"]:
            self.add_real_precipitaion(atm)
        else:
            self.add_fake_precipitation(atm)

        # ml.add_atmospheric_bc(atm)
        ml.add_atmospheric_bc(atm)

    def add_materials(self):
        ml = self.ml
        m = ml.get_empty_material_df(n=1)
        # [residual water content, saturated water content, a, n, hydraulic conductivity, l], [4 paramters of nitrate transport]
        m.loc[1] = [self.dynamic_config["resid_wc"], self.dynamic_config["sat_wc"], self.dynamic_config["alpha"], self.dynamic_config["n_empiric"], self.dynamic_config["h_conductivity"], self.static_config["l"], *self.static_config["nitrate_trans"]] # 6 retention curve + 4 soil solute paramters $sand
        ml.add_material(m)

    def add_solute(self):
    # =============================================================================
    # solute 
    # =============================================================================
        ml = self.ml
        sol1 = ml.get_empty_solute_df()
        sol1["beta"] = self.static_config["sol_beta"]
        ml.add_solute(sol1, difw=self.static_config["sol_difw"])

    def create_profile(self):
        ml = self.ml
        profile = ps.create_profile()
        profile['h'] = self.static_config["inital_wc"]
        profile['Conc'] = self.static_config["initial_conc"]
        root_distribution = self.static_config["root_distribution"](self.dynamic_config["root_depth"])
        profile['Beta'] = self.static_config["root_distribution_fill"](root_distribution, profile) # define root distribution in profile df
        ml.add_profile(profile)

    
    def initialize_model(self):
        ml = self.ml
        ml.add_time_info(tmax=self.static_config["n_hours"], print_times=True) #,dt=10^(-3), dtmin=10^(-7), dtmax=10^(-2))
        ml.add_waterflow(model=self.static_config["VAN_GENUCH_6_PARAM"],top_bc=self.static_config["ATM_W_SURFACE_RUNOFF"], bot_bc=self.static_config["FREE_DRAINAGE"], linitw=True)
        self.add_materials()
        self.create_profile()
        ml.add_obs_nodes(self.static_config["DEPTHS"]) # to check if possible to add two depth 10 and 20 cm
        ml.add_solute_transport(model=self.static_config["EQ_SOLUTE_TRANPORT"], top_bc=self.static_config["CAUCHY_BOUNDRY_COND"], bot_bc=self.static_config["CONT_CONC_PROFILE"]) #equilibrium model, upper BC - conc flux BC, lower BC = zero conc gradient. 
        self.add_atm_pressure()
        self.add_solute()
        ml.add_root_uptake(model=self.static_config["FEDES_ET_AL"], crootmax=self.dynamic_config["root_depth"], p0=self.static_config["p0"], p2h=self.static_config["p2h"], p2l=self.static_config["p2l"], p3=self.static_config["p3"], r2h=self.static_config["r2h"], r2l=self.static_config["r2l"], poptm=self.static_config["poptm"]) # model=Feddes, define Cmax, paramters for tomato from hydrus library 
        ml.write_input()
        print("MODEL INITIALIZED")

    def get_cvRoot(self):
        # sum(cvRoot) == comulative solute uptake
        # sum(cvBot) == comulative bottom flux solute
        ml = self.ml
        solute_levels = ml.read_solutes()
        print(solute_levels)
        return solute_levels[["Sum(cvRoot)"]] 

    def pretty_show_model(self):
        """forward simulation"""
        ml = self.ml
        self.run_model()
        df = ml.read_tlevel()
        df.plot(subplots=True)
        plt.show()

    def run_model(self):
        ml = self.ml
        start_execution = time.time()
        ml.simulate()
        end_execution = time.time()
        print('run time is:' + str(end_execution-start_execution))