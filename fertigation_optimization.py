
#%%
import os

import pandas as pd
import numpy as np
import phydrus as ps
import matplotlib.pyplot as plt
import time
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
inital_wc = 0.08
initial_conc = 0
croot_max = 40

# fertigation paramters
fertigation_conc = 40 # N-NO3
daily_et = 0.5 # output flux of water by evaporation or by plant uptake 

leaching_fraction = 1.1
irrigation = daily_et*leaching_fraction
convertion_factor = 10 # between [mg/l * cm] as an input to [mg/m2] as the desired output
n_days = 30 # choose between 1 to 31
t_max = 24*n_days

applied_N = irrigation* fertigation_conc* n_days
atm_columns = ["tAtm", "Prec", "rSoil", "rRoot", "hCritA", "rB", "hB", "ht", "tTop", "tBot", "Ampl", "cTop", "cBot"]

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

def add_atm_pressure(ml):
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

    precipitation_interval = [5+(i*24) for i in range(n_days)]
    atm.iloc[[precipitation_interval], 1] = irrigation
    atm.iloc[[precipitation_interval], 11] = fertigation_conc #ctop

    # ml.add_atmospheric_bc(atm)
    ml.add_atmospheric_bc(atm)

def add_materials(ml, h_conductivity):
    m = ml.get_empty_material_df(n=1)
    # [residual water content, saturated water content, a, n, hydraulic conductivity, l], [4 paramters of nitrate transport]
    m.loc[1] = [0.045, 0.3, 0.145, 2.68, h_conductivity, -0.5, 1.5, 10, 1, 0] # 6 retention curve + 4 soil solute paramters $sand
    # m.loc[1] = [0.065, 0.41, 0.075, 1.89, 4.42, -0.5, 1.5, 10, 1, 0] # 6 retention curve + 4 soil solute paramters $sandy loam
    # m.loc[1] = [0.057, 0.41, 0.124, 2.28, 14.597, -0.5, 1.5, 10, 1, 0] # 6 retention curve + 4 soil solute paramters $loamy sand
    # m.loc[1] = [0.078, 0.43, 0.036, 1.56, 1.04, -0.5, 1.5, 10, 1, 0] # 6 retention curve + 4 soil solute paramters $loam

    ml.add_material(m)

def add_solute(ml):
    # =============================================================================
    # solute 
    # =============================================================================
    sol1 = ml.get_empty_solute_df()
    sol1["beta"] = 1
    ml.add_solute(sol1, difw=0.068, difg=0)

def create_profile(ml):
    profile = ps.create_profile(top=0,bot=-100, h=0.12, dx=0.1, conc=0)
    profile['h'] = inital_wc
    profile['Conc'] = initial_conc
    root_dist1 = np.linspace(1,0,10) # root distribution decrease linearly from 1 to 0 in the upper 30 cm
    root_dist2 =  np.zeros(len(profile)-len(root_dist1)) # root distribution is 0 from 30 to 100 cm
    profile['Beta'] = np.concatenate([root_dist1,root_dist2]) # define root distribution in profile df

    ml.add_profile(profile)

def initialize_model(ml, h_conductivity):
    ml.add_time_info(tinit=0, tmax=t_max, print_times=True,dt=10^(-3), dtmin=10^(-7), dtmax=10^(-2))
    ml.add_waterflow(model=0,top_bc=3, bot_bc=4, linitw=True)
    ml.add_solute_transport(model=0, top_bc=0, bot_bc=0) #equilibrium model, upper BC - conc flux BC, lower BC = zero conc gradient. 
    add_materials(ml, h_conductivity)
    create_profile(ml)
    ml.add_obs_nodes([20]) # to check if possible to add two depth 10 and 20 cm
    add_atm_pressure(ml)
    add_solute(ml)
    ml.add_root_uptake(model=0, crootmax=croot_max, p0=- 10, p2h=- 800, p2l=-1500, p3=- 8000, r2h=0.02, r2l=0.004, poptm=[-25]) # model=Feddes, define Cmax, paramters for tomato from hydrus library 
    ml.write_input()

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

def get_cvRoot_cvBot(ml):
    # sum(cvRoot) == comulative solute uptake
    # sum(cvBot) == comulative bottom flux solute
    solute_levels = ml.read_solutes()
    print(solute_levels)
    return solute_levels[["Sum(cvRoot)", "Sum(cvBot)"]] 

#def create_graph_with_legen(x_axis, y_axis, legend):



def create_h_conductivity_table():
    final_df = pd.DataFrame()
    for h_conductivity in range(10, 100, 10):
        ml = main(h_conductivity)
        df = get_cvRoot_cvBot(ml)
        df["h_conductivity"] = h_conductivity
        df["applied_N"] = applied_N
        if final_df.empty:
            final_df = df
        else:
            final_df = pd.concat([final_df, df])
    
    print(final_df)
    for column in final_df.columns:
        print("This:" + column + "is column name.")
    print(final_df.index)


    # Plotting cv_root vs time
    plt.figure(figsize=(10, 5))
    for h_conductivity, group_df in final_df.groupby('h_conductivity'):
        plt.plot(group_df.index, group_df['Sum(cvRoot)'], label=f'h_conductivity {h_conductivity}')
    
    plt.xlabel('Time')
    plt.ylabel('Sum(cvRoot)')
    plt.title('Sum(cvRoot) over Time for Different h_conductivity Values')
    plt.legend(title='h_conductivity')
    plt.show()
    
    # Plotting Sum(cvBot) vs time
    plt.figure(figsize=(10, 5))
    for h_conductivity, group_df in final_df.groupby('h_conductivity'):
        plt.plot(group_df.index, group_df['Sum(cvBot)'], label=f'h_conductivity {h_conductivity}')
    plt.xlabel('Time')
    plt.ylabel('Sum(cvBot)')
    plt.title('Sum(cvBot) over Time for Different h_conductivity Values')
    plt.legend(title='h_conductivity')
    plt.show()

    return final_df





    """final_df.plot(subplots=True)
    plt.legend(title='h_conductivity')
    plt.show()

    return final_df"""

#%%
# =============================================================================
# # this script is calculating Nitrogen use efficency as a function Croot max

# def hydrus_model(p): #p[0] is irrigation p[1] is fertilization
# input units: cm, ppm, please note to unit convertion in the output of the script
def main(h_conductivity=30): #
    exe = os.path.join(os.getcwd(), "hydrus.exe") #this never change!!
    # Create the basic model
    ml = ps.Model(exe_name=exe, ws_name=ws, name="model", description=desc,
                mass_units="M", time_unit="hours", length_unit="mm")
    #initialize the model
    initialize_model(ml, h_conductivity)
    run_model(ml)
    return ml

if __name__ == "__main__":
    print(create_h_conductivity_table())

