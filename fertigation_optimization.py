
#%%
import os

import pandas as pd
import numpy as np
import phydrus as ps
import matplotlib.pyplot as plt
import time
from collections import UserDict
from typing import List
#%%
# =============================================================================
# This script is running hydrus model for 30 days
# with daily fertigation while checking the soil concentration and NUE
# relates to different Croot max value
# a term called relative Croot max is the ratio between the estimated C root max and the real C root max
# when N is applied in the concentration of the estimated Croot max
# 
# a function called 'hydrus model' run the model, all the paramters of ET, irrigation, soil type and etc
# should be define inside it
# 
# the C root max value is a paramter of the function and is running as a for loop for different values
# =============================================================================

ws = "fert_optimization"
desc = "Fertilization optimization"

#some initial paramters:
inital_wc = 0.1
initial_conc = 0
croot_max = 40

# fertigation paramters
fertigation_conc = 40 # N-NO3
daily_et = 0.5 # output flux of water by evaporation or by plant uptake 


convertion_factor = 10 # between [mg/l * cm] as an input to [mg/m2] as the desired output
n_days = 30 # choose between 1 to 31
t_max = 24*n_days

atm_columns = ["tAtm", "Prec", "rSoil", "rRoot", "hCritA", "rB", "hB", "ht", "tTop", "tBot", "Ampl", "cTop", "cBot"]

def calculate_applied_N(config):
    return daily_et * config["leaching_fraction"] * fertigation_conc * n_days

class Config(UserDict):
    # Example usage
    default_config = {
        "h_conductivity" : 1,
        "resid_wc" : 0.075,
        "sat_wc" : 0.3,
        "alpha" : 0.1075,
        "n_empiric" : 2.285,
        "root_depth" : 20,
        "leaching_fraction" : 1
    }

    def __init__(self, defaults=default_config):
        super().__init__(defaults)


"""
Distributes the daily evapotranspiration value into hourly values based on a parabolic distribution.
between 7 am to 17 pm, where tranpiration is 0.9 ET and evaporation is 0.1

Parameters:
- daily_evapotranspiration (float): The daily evapotranspiration value.
- n (int): Number of repetitions (days) for which the hourly distribution is to be generated. Default is 1.

Returns:
- pd.DataFrame: A DataFrame with columns 'hour', 'evapotranspiration', 'transpiration', and 'evaporation'.
"""
def linear_distribute_ET(daily_ET, n=1):
    ET = np.zeros(24)
    ET[7:18] = daily_ET/(18-7)
    transpiration = 0.9 * ET
    evaporation = 0.1 * ET
    hours = np.arange(24*n)
    # Create the DataFrame with repeated values
    return pd.DataFrame({
        'hour': hours,
        'evapotranspiration': np.tile(ET, n),
        'transpiration': np.tile(transpiration, n),
        'evaporation': np.tile(evaporation, n)
    })

def add_atm_pressure(ml, config):
    # (module) -> None
    # =============================================================================
    # atm_bc
    # =============================================================================
    ET = linear_distribute_ET(daily_et, n_days)
    atm = pd.DataFrame(0, index=np.arange(t_max), columns=atm_columns) # add columns according to the defined n_days

    atm['tAtm'] = np.arange(1,t_max+1)
    atm['rSoil'] = ET['evaporation']
    atm['rRoot'] = ET['transpiration']
    atm['hCritA'] = 1000000
    irrigation = daily_et*config["leaching_fraction"]

    precipitation_interval = [5+(i*24) for i in range(n_days)]
    atm.iloc[[precipitation_interval], 1] = irrigation
    atm.iloc[[precipitation_interval], 11] = fertigation_conc #ctop

    # ml.add_atmospheric_bc(atm)
    ml.add_atmospheric_bc(atm)

def add_materials(ml, config):
    m = ml.get_empty_material_df(n=1)
    print("H_conductivty: " + str(config["h_conductivity"]))
    # [residual water content, saturated water content, a, n, hydraulic conductivity, l], [4 paramters of nitrate transport]
    m.loc[1] = [config["resid_wc"], config["sat_wc"], config["alpha"], config["n_empiric"], config["h_conductivity"], -0.5, 1.5, 10, 1, 0] # 6 retention curve + 4 soil solute paramters $sand

    ml.add_material(m)

def add_solute(ml, config):
    # =============================================================================
    # solute 
    # =============================================================================
    sol1 = ml.get_empty_solute_df()
    sol1["beta"] = 1
    ml.add_solute(sol1, difw=0.068, difg=0)

def create_profile(ml, config):
    profile = ps.create_profile(top=0,bot=-100, h=0.12, dx=0.1, conc=0)
    profile['h'] = inital_wc
    profile['Conc'] = initial_conc
    root_dist1 = np.linspace(1,0,int(config["root_depth"])) # root distribution decrease linearly from 1 to 0 in the upper 30 cm
    root_dist2 =  np.zeros(len(profile)-len(root_dist1)) # root distribution is 0 from 30 to 100 cm
    profile['Beta'] = np.concatenate([root_dist1,root_dist2]) # define root distribution in profile df

    ml.add_profile(profile)

def initialize_model(ml, config):
    ml.add_time_info(tinit=0, tmax=t_max, print_times=True) #,dt=10^(-3), dtmin=10^(-7), dtmax=10^(-2))
    ml.add_waterflow(model=0,top_bc=3, bot_bc=4, linitw=True)
    ml.add_solute_transport(model=0, top_bc=-1, bot_bc=0) #equilibrium model, upper BC - conc flux BC, lower BC = zero conc gradient. 
    add_materials(ml, config)
    create_profile(ml, config)
    ml.add_obs_nodes([20]) # to check if possible to add two depth 10 and 20 cm
    add_atm_pressure(ml, config)
    add_solute(ml, config)
    ml.add_root_uptake(model=0, crootmax=croot_max, p0=-10, p2h=-400, p2l=-600, p3=-8000, r2h=0.02, r2l=0.004, poptm=[-25]) # model=Feddes, define Cmax, paramters for tomato from hydrus library 
    ml.write_input()
    print("MODEL INITIALIZED")

def pretty_show_model(ml):
    """forward simulation"""
    run_model(ml)
    df = ml.read_tlevel()
    df.plot(subplots=True)
    plt.show()

def run_model(ml):
    start_execution = time.time()
    ml.simulate()
    end_execution = time.time()
    print('run time is:' + str(end_execution-start_execution))

def get_cvRoot(ml):
    # sum(cvRoot) == comulative solute uptake
    # sum(cvBot) == comulative bottom flux solute
    solute_levels = ml.read_solutes()
    print(solute_levels)
    return solute_levels[["Sum(cvRoot)"]] 

def create_graph_with_legend(df, y_axis, legend):
    #nassuming the x axis is the df.index
    # Plotting cv_root vs time
    plt.figure(figsize=(10, 5))
    for legend_val, group_df in df.groupby(legend):
        plt.plot(group_df.index, group_df[y_axis], label=legend+f'{legend_val}')
    
    plt.xlabel('Time') # default
    plt.ylabel(y_axis)
    plt.title(f'{y_axis} over Time for Different {legend} Values')
    plt.legend(title=legend)
    plt.show()

def create_graph(df: pd.DataFrame, x_axis: str, y_axis: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_axis], df[y_axis])
    plt.title(f'{y_axis} vs {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.show(block=False)

def extract_relevant_data(ml : ps.Model, config : Config) -> float:
    """
    Use case specific extracts the information from the module of the output variable
    """
    df = get_cvRoot(ml)
    df["applied_N"] = calculate_applied_N(config)
    df["NUE"] = df["Sum(cvRoot)"] / df["applied_N"]
    # extracting the NUE from the latest available time, this will be the NUE value to represent this var_value
    max_time_row = df.loc[df.index.max()] 
    nue_for_var = max_time_row["NUE"]
    return nue_for_var
    
def add_relevant_data(final_df : pd.DataFrame, relevant_data : float, input_variable : str, var_value : float, output_variable : str) -> None:
    """
    Adds whatever information from the extract_relevant_data function to the final dataframe
    """
    print(f"{input_variable}: {var_value}, {output_variable}: {relevant_data}")
    return final_df.append({input_variable: var_value, output_variable : relevant_data}, ignore_index=True)

def create_variable_table(input_variable : str, output_variable : str, start_val : float, end_val : float, jump_val : float) -> pd.DataFrame: 
    """
    Function gets an input variable and output variable and a range in which we iterate over the input variable
    The function plots a graph of the output variable as a function of the input varibale and returns a dataframe with 
    with the information as well
    """
    config = Config()
    final_df = pd.DataFrame({input_variable : [], output_variable : []})
    for var_value in np.arange(start_val, end_val, jump_val):
        var_value = round(var_value, 2)
        config[input_variable]= var_value
        ml = main(config)
        relevant_data = extract_relevant_data(ml, config)
        print(relevant_data)
        final_df = add_relevant_data(final_df, relevant_data, input_variable, var_value, output_variable)
    final_df.to_csv(f"{input_variable}_table.csv")
    create_graph(final_df, input_variable, output_variable)

    return final_df



#%%
# =============================================================================
# # this script is calculating Nitrogen use efficency as a function Croot max

# def hydrus_model(p): #p[0] is irrigation p[1] is fertilization
# input units: cm, ppm, please note to unit convertion in the output of the script
def main(config): #
    exe = os.path.join(os.getcwd(), "../hydrus.exe") #this never change!!
    # Create the basic model
    ml = ps.Model(exe_name=exe, ws_name=ws, name="model", description=desc,
                mass_units="M", time_unit="hours", length_unit="mm")
    #initialize the model
    initialize_model(ml, config)
    run_model(ml)
    return ml

def create_variable_graphs():
    print(create_variable_table("h_conductivity", "NUE", 1, 20, 2))
    print(create_variable_table("resid_wc", "NUE", 0.05, 0.1, (0.1-0.05)/10))
    print(create_variable_table("sat_wc", "NUE", 0.2, 0.4, (0.4-0.2)/10))
    print(create_variable_table("alpha", "NUE", 0.075, 0.14, (0.14-0.075)/10))
    print(create_variable_table("n_empiric", "NUE", 1.89, 2.68, (2.68-1.89)/10))
    print(create_variable_table("root_depth", "NUE", 10, 30, (30-10)/10))
    print(create_variable_table("leaching_fraction", "NUE", 0.8, 1.2, (1.2-0.8)/10))

def create_graphs_from_csv(filenames : List[str] = ["h_conductivity_table.csv", "leaching_fraction_table.csv", "n_empiric_table.csv", "resid_wc_table.csv", "root_depth_table.csv", "sat_wc_table.csv"]) -> None:
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axs = axs.flatten()

    # Loop through each file and plot
    for i, file in enumerate(filenames):
        df = pd.read_csv(file)
         # Use the first column as x-axis and second column as y-axis
        x_axis = df.columns[1]
        y_axis = df.columns[2]
        
        df.plot(x=x_axis, y=y_axis, ax=axs[i], title=f'{x_axis} vs {y_axis}')

    # Hide any empty subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    create_graphs_from_csv()

