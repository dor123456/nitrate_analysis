
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
from static_configuration import static_config
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

def create_variable_table(input_variable : str, output_variable : str, start_val : float, end_val : float, jump_val : float) -> pd.DataFrame: 
    """
    Function gets an input variable and output variable and a range in which we iterate over the input variable
    The function plots a graph of the output variable as a function of the input varibale and returns a dataframe with 
    with the information as well
    """
    dynamic_config = DynamicConfig()
    final_df = pd.DataFrame({input_variable : [], output_variable : []})
    for var_value in np.arange(start_val, end_val, jump_val):
        var_value = round(var_value, 2)
        dynamic_config[input_variable]= var_value
        phy_ml = PhydrusModelInit(dynamic_config, static_config)
        relevant_data = extract_relevant_data(phy_ml)
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
    create_variable_graphs()

