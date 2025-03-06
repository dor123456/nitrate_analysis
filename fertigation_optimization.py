
#%%
import os

import pandas as pd
import numpy as np
import phydrus as ps
import matplotlib.pyplot as plt
import time
from collections import UserDict
from typing import List
from phydrus_model_initializer import DynamicConfig, PhydrusModelInit
# from static_configuration import static_config
from static_configuration import static_config
from scipy.optimize import minimize, least_squares
import io
import sys

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


def create_graph_with_legend(df, y_axis, legend):
    # assuming the x axis is the df.index
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

def extract_relevant_data(phy_ml : PhydrusModelInit) -> float:
    """
    Use case specific extracts the information from the module of the output variable
    """
    df = phy_ml.get_cvRoot()
    df["applied_N"] = phy_ml.get_applied_N()
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

def create_variable_table(input_variable : str, output_variable : str, start_val : float, end_val : float, num_of_jumps : float=10) -> pd.DataFrame: 
    """
    Function gets an input variable and output variable and a range in which we iterate over the input variable
    The function plots a graph of the output variable as a function of the input varibale and returns a dataframe with 
    with the information as well
    """
    dynamic_config = DynamicConfig()
    final_df = pd.DataFrame({input_variable : [], output_variable : []})
    for var_value in np.linspace(start_val, end_val, num_of_jumps):
        var_value = round(var_value, 2)
        dynamic_config[input_variable]= var_value
        phy_ml = PhydrusModelInit(dynamic_config, static_config)
        relevant_data = extract_relevant_data(phy_ml)
        print(relevant_data)
        final_df = add_relevant_data(final_df, relevant_data, input_variable, var_value, output_variable)
    final_df.to_csv(f"variable_csvs/{input_variable}_table.csv")
    create_graph(final_df, input_variable, output_variable)

    return final_df



#%%
# =============================================================================
# # this script is calculating Nitrogen use efficency as a function Croot max

# def hydrus_model(p): #p[0] is irrigation p[1] is fertilization
# input units: cm, ppm, please note to unit convertion in the output of the script

def create_variable_graphs():
    print(create_variable_table("h_conductivity", "NUE", 1, 20))
    print(create_variable_table("resid_wc", "NUE", 0.05, 0.1))
    print(create_variable_table("sat_wc", "NUE", 0.2, 0.4))
    print(create_variable_table("alpha", "NUE", 0.075, 0.14))
    print(create_variable_table("n_empiric", "NUE", 1.89, 2.68))
    print(create_variable_table("root_depth", "NUE", 10, 30))
    print(create_variable_table("leaching_fraction", "NUE", 0.8, 1.2))

def create_graphs_from_csv(filenames : List[str] = ["h_conductivity_table.csv", "leaching_fraction_table.csv", "n_empiric_table.csv", "resid_wc_table.csv", "root_depth_table.csv", "sat_wc_table.csv"]) -> None:
    """
    gets the filenames of relevant tables and plots them in pretty graphs
    """
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axs = axs.flatten()

    # Loop through each file and plot
    for i, file in enumerate(filenames):
        df = pd.read_csv("variable.csvs/" +file)
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


test_theta_interval = [f"11-{i}" for i in range(19, 31)] # 19 to 30 of november
wc_filename = "cleaned_real_water_content_data.csv"

def get_real_world_theta(depths=static_config["DEPTHS"], filename=wc_filename):
    """
    returns a dict between depths and a df with theta
    """
    real_wc = pd.read_csv(filename)
    depth_to_theta = {}
    for depth in depths:
        real_wc_for_depth  = real_wc[f"wc_B{abs(depth)}_mean"]
        real_wc_for_depth.index  = real_wc_for_depth.index + 1 # start indexes from 1 not 0 to be like the model
        depth_to_theta[depth] = real_wc_for_depth
    print("DEPTH: ")
    print(depth_to_theta)
    return depth_to_theta
    
def residual_function(model_wc, real_wc):
    """
    calculates the residual cost for all the data points' theta columns
    for different depths, between the model and the real world
    """
    print("REAL WC: ", real_wc)
    different_depths_resids = []
    for depth in model_wc.keys():
        if(len(model_wc[depth]) != len(real_wc[depth])):
            return [1000000000]
        print(f"MODEL WC FOR DEPTH {depth}: ", model_wc[depth])
        print(f"REAL WC FOR DEPTH {depth}:", real_wc[depth])
        depth_resids = np.abs(model_wc[depth] - real_wc[depth])
        print("DEPTH_RESIDS: ", depth_resids)
        different_depths_resids.append(depth_resids)
    resid = np.concatenate(different_depths_resids)
    return resid
def cost_function(model_wc, real_wc):
    """
    calculates the sum of the distances between the theta columns for different depths
    """
    resids = residual_function(model_wc, real_wc)
    # print(resids)
    return np.sum(resids)

def valid_input_range(params):
    """
    validates the input that it is in allowed range
    params[0] = alpha
    params[1] = n_empiric
    """
    if 0.036 <= params[0] and params[0] <= 0.145 and 1.37 <= params[1] and params[1] <= 2.68:
        return True
    return False

def minimize_resid_function(params):
    """
    the resid function the algorithm is supposed to minimize. Receives the parameters of alpha and n_empiric and
    outputs a list of all the resids between the theta in the real world and in the model
    """
    dynamic_config = DynamicConfig()
    dynamic_config['alpha'] = params[0]
    dynamic_config['n_empiric'] = params[1]
    if not valid_input_range(params):
        return 100000000
    phy_ml = PhydrusModelInit(dynamic_config, static_config)
    resids = residual_function(phy_ml.get_node_info(column_name="theta"), get_real_world_theta())
    cost = np.sum(resids)
    with open('params_cost_hydro_10.txt', 'a') as fd:
        fd.write('params: ' + str(params) + " cost: " + str(cost) + "\n")
    return resids

def minimize_cost_function(params):
    """
    cost function that recieves the params n and alpha and returns the sum of the distances between
    the theta in all depths in the model and the theta in all the depths in the real world.
    """
    resids = minimize_resid_function(params)
    print(resids)
    return np.sum(resids)

def get_solute_variables():
    """
    The function that runs the logic behing inverse solution for the soil properties in the midrasha.
    """
    open("params_cost_hydro_10.txt", "w").close() # erase the ouput file where we write the params and their cost
    initial_params_dict = {"alpha" : 0.11560592, # 0.036 <= x <= 0.145
        "n_empiric" : 1.764367} # 1.37 <= x <=2.68   
    result = minimize(minimize_cost_function, list(initial_params_dict.values()),method="Nelder-Mead",
    options={'disp': True, 'maxiter': 1000})
    # result = least_squares(minimize_resid_function, list(initial_params_dict.values()),method="lm",
    # verbose=2)
    optimized_params_dict = dict(zip(initial_params_dict.keys(), result.x))
    print(optimized_params_dict)
    return optimized_params_dict

if __name__ == "__main__":
    get_solute_variables()
    # dynamic_config = DynamicConfig()
    # phy_ml = PhydrusModelInit(dynamic_config, static_config)
    # df = phy_ml.get_cvRoot()
    # print(df)
    # phy_ml.ml.read_nod_inf()
    # print(phy_ml.get_theta())
    # phy_ml.get_theta()
    # get_real_world_theta()
    # create_variable_graphs()
    # create_graphs_from_csv()"""
    # minimize_resid_function([0.124,2.28])

    


