"""
general idea- take initial parameters in good format run a phydrus model with these parameters. 
Take the output, use the output fertigation concentration and any other relevant parameters we will find save it in a file
and send it to the function that decides how much fertigation to give. Use the parameters outputed from the model 
and the amount of fertigation to set initial parameters to run the next step. 

1) initial parameters- make a list of the parameters, take the static config and change it directly
2) function that extracts information from run model, changes the static config and saves relevent data for the AI model in a file
3) function that reads the data from the file and uses AI model to make a prediction
4) function 

"""

from simple_pid import PID
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from phydrus_model_initializer import DynamicConfig, PhydrusModelInit
from static_configuration import static_config


def grow_linear_stay_at_max(min, max, step_until_max, length):
        """
        Returns a numpy array that grows linearly from min to max for 
        step_until_max steps and then stays at the max
        """
        return np.concatenate((np.linspace(min, max, step_until_max), np.full(length-step_until_max, max)))

def transpiration_calculation(LAI, k=1):
    """
    Returns the transpiration frac from the LAI and k
    """
    return 1 - math.exp(-k*LAI)

def get_ET_values(filepath="ET_Ovdat.csv", column_name="ET"):
    # Read the Csv file into a DataFrame
    df = pd.read_csv(filepath)

    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the Excel file.")

    df[column_name] = df[column_name] / 10 # convert mm to cm

    # Return the values of the specified column as a list
    return np.array(df[column_name].tolist())

class NitrateOptimizer():
    past_days_data_file = "water_content_and_fertigation_history.csv"
    static_configuration = None
    dynamic_configuration = None
    current_step = 0
    max_step = 240
    phydrus = None
    pid = None
    irigation_fertigation_log = []
    plant_growing_time = 40
    big_change_counter = 5
    alpha_nit = 0.15
    fertigation = 20
    simulation_changing_params = {"croot_max_list" : grow_linear_stay_at_max(0, 100, plant_growing_time, max_step+1), #croot max gradually increasing in first plant growing time days and then stays at max
                                  "transpiration_frac_list" : np.array([transpiration_calculation(LAI) for LAI in grow_linear_stay_at_max(0, 1, plant_growing_time, max_step+1)]),
                                  "ET_list" : get_ET_values(filepath="ET_Ovdat_long.csv")}
                                  
    def __init__(self, static_configuration = static_config, dynamic_configuration = DynamicConfig()):
        self.static_configuration = static_configuration
        self.dynamic_configuration = dynamic_configuration
        
        # Start out pid algorithm for choosing fertigation levels
        self.pid = PID(Kp=0.4, Ki=0.07, Kd=0.25, setpoint=40.0)
        self.pid.output_limits = (-100, 100)  # Limit fertigation between 0 and 10 liters


    def clear_past_days_data(self):
        # Read the header of the CSV file without loading the data
        df = pd.read_csv(self.past_days_data_file)

        # Save only the headers back to the file (overwrite the file with only headers)
        df.head(0).to_csv(self.past_days_data_file, index=False)  # head(0) returns only the header


    def run(self):
        """
        Runs the whole simulation from start to finish after inisialization
        """
        self.clear_past_days_data()
        self.update_config_changing_params(0)

        while self.current_step < self.max_step:
            self.step()
            self.current_step += 1

        self.plot_nitrate_concentration()
        print(self.irigation_fertigation_log)

    def step(self):
        """
        Runs one step out of the simulation
        """
        phydrus= PhydrusModelInit(self.dynamic_configuration, self.static_configuration)
        self.phydrus = phydrus
        self.update_init_params()
        self.update_past_data_file()
        self.compute_cost_function()
        irrigation_fertigation = self.decide_irrigation_and_fertigation()
        self.irigation_fertigation_log.append(irrigation_fertigation)

    def compute_cost_function(self):
        return 0
    
    def update_config_changing_params(self, step):
        """
        Updates the static configuration by the changing params for the step given
        (The params that dont rely on yesterday params but change through time) 
        """
        self.static_configuration["transpiration_frac"] = self.simulation_changing_params["transpiration_frac_list"][step]
        self.static_configuration["croot_max"] = self.simulation_changing_params["croot_max_list"][step]
        self.static_configuration["daily_et"] = self.simulation_changing_params["ET_list"][step]
        print(self.static_configuration["transpiration_frac"])
        print(self.static_configuration["croot_max"])
        print(self.static_configuration["daily_et"])

    def update_init_params(self):
        """
        update the next steps initial params based on the previous steps output
        """
        node_info = self.phydrus.ml.read_nod_inf(times=[self.static_configuration["n_hours"]])
        self.static_configuration["initial_wc_distribution"] = node_info["Moisture"]
        self.static_configuration["initial_conc_distribution"] = node_info["Conc(1..NS)"]
        print("Initial Water Content Distribution (Moisture):", self.static_configuration["initial_wc_distribution"])
        print("\nInitial Concentration Distribution:", self.static_configuration["initial_conc_distribution"])
        self.static_configuration["auto_wc_and_NO3"] = False
        self.update_config_changing_params(self.current_step+1) # the changing params of next step

    def update_past_data_file(self):
        """
        Save the important information to our csv log in past_days_data_file
        """
        water_content = self.phydrus.get_node_info(column_name="theta")
        concentration = self.phydrus.get_node_info(column_name='Conc')
        soil_data = {
            "Step" : [self.current_step],
            "WC_10" : [water_content[-10].iloc[-1]],
            "WC_20" : [water_content[-20].iloc[-1]],
            "WC_40" : [water_content[-40].iloc[-1]],
            "Conc_10" : [concentration[-10].iloc[-1]],
            "Conc_20" : [concentration[-20].iloc[-1]],
            "Conc_40" : [concentration[-40].iloc[-1]]
        }
        df = pd.DataFrame(soil_data)
    
        # Append to existing file assuming there are column headers already
        df.to_csv(self.past_days_data_file, mode='a', header=False, index=False)

    def load_past_data(self, n_days=3):
        """
        Load the past `n_days` of HYDRUS simulation data.
        
        :param n_days: Number of previous days to consider for decision-making
        :return: DataFrame with past `n_days` of data
        """
        if not os.path.exists(self.past_days_data_file):
            return None  # No past data available

        df = pd.read_csv(self.past_days_data_file)

        # Get last `n_days` of data
        return df.tail(n_days)

    def plot_nitrate_concentration(self, goal_column="Conc_10", goal_value=40):
        """
        Plots nitrate concentration over time at a given soil depth.
        
        Parameters:
        - csv_file (str): Path to the CSV file.
        - depth (int): Depth in cm (default: -20 cm).
        - goal (float): Target nitrate concentration (default: 40 ppm).
        """
        # Load the CSV file
        df = pd.read_csv(self.past_days_data_file)

        # Extract nitrate concentration at the given depth
        if goal_column not in df.columns:
            raise ValueError(f"Goal column {goal_column} not found in CSV columns.")

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(df["Step"], df[goal_column], label=f"Nitrate at {goal_column} cm", marker="o")
        
        # Add the goal line
        plt.axhline(goal_value, color='r', linestyle='--', label=f"Goal: {goal_value} ppm")
        
        # Labels and title
        plt.xlabel("Time (days)")
        plt.ylabel("Nitrate Concentration (ppm)")
        plt.title(f"Nitrate Concentration at {goal_column} Over Time")
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def decide_irrigation_and_fertigation(self):
        past_data = self.load_past_data(1)  # Use last 3 days for decision-making
        # Every 60 days set the precipitation to be 1.5 times that day's ET to count for different seasons
        precipitation = self.simulation_changing_params["ET_list"][self.current_step//60] * 1.5 
        calculated_error = self.pid(past_data["Conc_10"].iloc[0])
        self.fertigation = max(0,self.fertigation + calculated_error)
        self.dynamic_configuration["fertigation_conc"] = self.fertigation
        self.dynamic_configuration["precipitation"] = precipitation
        return precipitation, self.fertigation, calculated_error


if __name__ == "__main__":
    dynamic_config = DynamicConfig()
    dynamic_config["h_conductivity"] = 30
    dynamic_config["resid_wc"] = 0.04
    dynamic_config["sat_wc"] = 0.32
    dynamic_config["alpha"] = 0.15
    dynamic_config["n_empiric"] = 2.1
    simulation = NitrateOptimizer(dynamic_configuration=dynamic_config)
    simulation.run()