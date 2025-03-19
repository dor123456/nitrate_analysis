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

import copy
from static_configuration import static_config

from simple_pid import PID
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from phydrus_model_initializer import DynamicConfig, PhydrusModelInit


def estimate_daily_nitrogen_uptake(total_days, total_nitrogen, max_timestep):
    """
    Estimate daily nitrogen uptake based on total growing time and cumulative nitrogen uptake,
    and ensure the maximum uptake rate is maintained after the growing period.

    Parameters:
    - total_days (int): Total growing time in days.
    - total_nitrogen (float): Total nitrogen absorbed during the growing period (e.g., mmol).
    - max_timestep (int): The total duration of the simulation (in days).

    Returns:
    - np.ndarray: Daily nitrogen uptake values for the entire simulation period.
    """

    # Sigmoid parameters to shape the uptake curve
    midpoint = total_days * 0.5  # Day of maximum growth rate (midpoint of the curve)
    growth_rate = 0.1            # Controls the steepness of the sigmoid curve

    # Generate days array (from 0 to max_timestep)
    days = np.arange(max_timestep)

    # Sigmoid function to model nitrogen uptake pattern for the growing period
    uptake_fraction = 1 / (1 + np.exp(-growth_rate * (days[:total_days] - midpoint)))

    # Normalize to ensure the total uptake sums to total_nitrogen for the growing period
    daily_uptake = (uptake_fraction / np.sum(uptake_fraction)) * total_nitrogen

    # After the growing period, maintain the maximum nitrogen uptake rate
    max_uptake = daily_uptake[-1]  # Last value of the growing period
    daily_uptake = np.concatenate([daily_uptake, np.full(max_timestep - total_days, max_uptake)])

    return daily_uptake
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
    goal_ppm = 10 # threshold value 
    goal_ppm = 10
    wc_goal = 0.16
    past_days_data_file = "water_content_and_fertigation_history.csv"
    max_step = 65 # number of days 
    plant_growing_time = 60 # growing period and than platue
    
    simulation_changing_params = {# "active_uptake_amount_list" : grow_linear_stay_at_max(0,700/plant_growing_time, plant_growing_time, max_step+1), # for linear incalantion in uptake and then stay at max
                                  "active_uptake_amount_list" : estimate_daily_nitrogen_uptake(plant_growing_time, 650, max_step+1), # for sigmoid function inclantion in uptake and then stay at max
                                  "croot_max_list" : grow_linear_stay_at_max(0, 100, plant_growing_time, max_step+1), #croot max gradually increasing in first plant growing time days and then stays at max
                                  "transpiration_frac_list" : np.array([transpiration_calculation(LAI) for LAI in grow_linear_stay_at_max(0, 1, plant_growing_time, max_step+1)]),
                                  "ET_list" : get_ET_values()} # filepath="ET_Ovdat_long.csv")}
                                  
    def __init__(self, static_configuration = static_config, dynamic_configuration = DynamicConfig(), static_fertigation=False):
        self.static_configuration = copy.deepcopy(static_configuration) # dont change original static config
        self.dynamic_configuration = dynamic_configuration
        self.static_fertigation = static_fertigation
        self.current_step = 0
        self.phydrus = None
        self.irrigation_fertigation_log = []
        self.fertigation = 40
        self.precipitation = 0.4
        # Start out pid algorithm for choosing fertigation levels
        self.pid = PID(Kp=0.4, Ki=0.07, Kd=0.625, setpoint=self.goal_ppm)
        self.pid.output_limits = (-100, 100)  # Limit fertigation between 0 and 10 liters
        self.water_pid = PID(Kp=1, Ki=0.1, Kd=0.05, setpoint=self.wc_goal)
        self.water_pid.output_limits = (-0.1,0.1)


    def clear_past_days_data(self):
        """
        Erase past simulation data
        """
        # Read the header of the CSV file without loading the data
        df = pd.read_csv(self.past_days_data_file)

        # Save only the headers back to the file (overwrite the file with only headers)
        df.head(0).to_csv(self.past_days_data_file, index=False)  # head(0) returns only the header


    def run(self):
        """
        Runs the whole simulation from start to finish after inisialization
        """
        self.clear_past_days_data()
        self.update_config_changing_params(0) # update the changing params for the first day

        while self.current_step < self.max_step:
            self.step()
            self.current_step += 1

        self.create_all_plots()
        print(self.irrigation_fertigation_log)

    def step(self):
        """
        Runs one step out of the simulation
        """
        phydrus= PhydrusModelInit(self.dynamic_configuration, self.static_configuration)
        self.phydrus = phydrus
        self.update_init_params()
        self.update_past_data_file()
        self.decide_irrigation()
        if not self.static_fertigation:
            self.decide_fertigation()
        irrigation_fertigation = (self.dynamic_configuration["precipitation"], self.dynamic_configuration["fertigation_conc"])
        self.irrigation_fertigation_log.append(irrigation_fertigation)
 
    def get_weekly_constant_et(self):
        """
        Returns the average ET of last week, in first week we take day 0 as last week
        """
        et_list = self.simulation_changing_params["ET_list"]
        # Identify the start of the previous week (rounded to full weeks)
        week_index = (self.current_step // 7) * 7
        start = max(0, week_index - 7)  # Ensure we don't go below index 0
        last_week_et = et_list[start:week_index]
        
        if len(last_week_et) == 0:
            return et_list[0]  # Fallback for the first week

        return np.mean(last_week_et)

    def decide_irrigation(self):
        # We take the ET of the past week on average and mutiply it by constant to get the amount of prcipitatio
        past_data = self.load_past_data(1)  # Use last 1 day for decision-making
        calculated_error = self.water_pid(past_data["WC_10"].iloc[0])
        print("calculated error ", calculated_error)
        self.precipitation = max(0.2,self.precipitation + calculated_error)
        self.dynamic_configuration["precipitation"] = self.precipitation
        print("Precipitation:", self.precipitation)
        return self.precipitation
        
    def decide_irrigation_by_et(self):
        et = self.get_weekly_constant_et()
        precipitation = et * 1.2
        self.dynamic_configuration["precipitation"] = precipitation 
        return precipitation

    
    def update_config_changing_params(self, step):
        """
        Updates the static configuration by the changing params for the step given
        (The params that dont rely on yesterday params but change through time in a predefined manner) 
        """
        self.static_configuration["transpiration_frac"] = self.simulation_changing_params["transpiration_frac_list"][step]
        # self.static_configuration["croot_max"] = self.simulation_changing_params["croot_max_list"][step]
        self.static_configuration["daily_et"] = self.simulation_changing_params["ET_list"][step]
        self.static_configuration["active_uptake_amount"] = self.simulation_changing_params["active_uptake_amount_list"][step] / 24 # spread equally on all hours
        

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
        cv_root = self.phydrus.get_cvRoot()
        print("cvroot last: " + str(cv_root))
        soil_data = {
            "Step" : [self.current_step],
            "WC_10" : [water_content[-10].iloc[-1]],
            "WC_20" : [water_content[-20].iloc[-1]],
            "WC_40" : [water_content[-40].iloc[-1]],
            "Conc_10" : [concentration[-10].iloc[-1]],
            "Conc_20" : [concentration[-20].iloc[-1]],
            "Conc_40" : [concentration[-40].iloc[-1]],
            "NO3_uptake" : [cv_root.iloc[-1]["Sum(cvRoot)"]]
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

    def decide_fertigation(self):
        """
        Use pid to calculate error by which we change the amount of fertilizer we give
        """
        past_data = self.load_past_data(1)  # Use last 1 day for decision-making
        calculated_error = self.pid(past_data["Conc_10"].iloc[0])
        self.fertigation = max(0,self.fertigation + calculated_error)
        self.dynamic_configuration["fertigation_conc"] = self.fertigation
        return self.fertigation
    
    def plot_water_content(self, goal_column="WC_10"):
        """
        Plots water content over time at a given soil depth.
        
        """
        goal_value = self.wc_goal
        # Load the CSV file
        df = pd.read_csv(self.past_days_data_file)

        # Extract nitrate concentration at the given depth
        if goal_column not in df.columns:
            raise ValueError(f"Goal column {goal_column} not found in CSV columns.")

        # Plot the data
        plt.figure(figsize=(10, 5))
        plt.plot(df["Step"], df[goal_column], label=f"Water Content at {goal_column} cm", marker="o")
        
        # Add the goal line
        plt.axhline(goal_value, color='r', linestyle='--', label=f"Goal: {goal_value} ppm")
        
        # Labels and title
        plt.xlabel("Time (days)")
        plt.ylabel("Water Content (fraction)")
        plt.title(f"Water Content at {goal_column} Over Time")
        plt.legend()
        plt.grid(True)


    def plot_nitrate_concentration(self, goal_column="Conc_10"):
        """
        Plots nitrate concentration over time at a given soil depth.
    
        """
        goal_value = self.goal_ppm # the goal ppm value is saved at the class level
        # Load the past data CSV file
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


    def plot_fertigation_graphs(self):
        # Define step indices
        steps = range(len(self.irrigation_fertigation_log))

        df = pd.read_csv(self.past_days_data_file)

        print("NO3_uptake: ", df["NO3_uptake"])

        cumulative_no3_uptake = np.cumsum(df["NO3_uptake"])
        # Calculate irrigation * fertigation values
        no3_applied = [irrigation * fertigation for irrigation, fertigation in self.irrigation_fertigation_log] # given in 10*mg/m**2 == (cm*m**2/m**2 * mg/((0.1m)**3))

        # Compute cumulative sum
        cumulative_no3_applied = np.cumsum(no3_applied)

        NUE = cumulative_no3_uptake/cumulative_no3_applied

        # Plotting
        fig, ax = plt.subplots(4, 1, figsize=(10, 8))

        # First graph: irrigation * fertigation
        ax[0].plot(steps, no3_applied, marker='o', linestyle='-', color='b', label='Irrigation × Fertigation')
        ax[0].set_title('Irrigation × Fertigation Over Time')
        ax[0].set_xlabel('Step')
        ax[0].set_ylabel('Irrigation × Fertigation (10 mg/m**2)')
        ax[0].legend()
        ax[0].grid()

        # Second graph: fertigation cumulative sum
        ax[1].plot(steps, cumulative_no3_applied, marker='o', linestyle='-', color='r', label='Cumulative Sum')
        ax[1].set_title('Cumulative Sum of Irrigation × Fertigation')
        ax[1].set_xlabel('Step')
        ax[1].set_ylabel('Cumulative Sum')
        ax[1].legend()
        ax[1].grid()

        # Third graph: NUE
        ax[2].plot(steps, NUE, marker='o', linestyle='-', color='g', label='NUE')
        ax[2].set_title('Nitrogen Use Efficiency (NUE) Over Time')
        ax[2].set_xlabel('Step')
        ax[2].set_ylabel('NUE (NO3 uptake / input)')
        ax[2].legend()
        ax[2].grid()

        # Forth graph CvRoot
        ax[3].plot(steps, cumulative_no3_uptake, marker='o', linestyle='-', color='purple', label='CvRoot')
        ax[3].set_title('Root Nitrogen uptake Over Time')
        ax[3].set_xlabel('Step')
        ax[3].set_ylabel('Root NO3 uptake')
        ax[3].legend()
        ax[3].grid()

        plt.tight_layout()


    def create_all_plots(self):
        self.plot_nitrate_concentration()
        self.plot_water_content()
        self.plot_fertigation_graphs()

def set_dynamic_config():
    dynamic_config = DynamicConfig()
    dynamic_config["h_conductivity"] = 3
    dynamic_config["resid_wc"] = 0.04
    dynamic_config["sat_wc"] = 0.32
    dynamic_config["alpha"] = 0.06
    dynamic_config["n_empiric"] = 1.6
    dynamic_config["fertigation_conc"] = 40
    dynamic_config["precipitation"] = 0.4
    return dynamic_config

if __name__ == "__main__":
    
    dynamic_config = set_dynamic_config()
    simulation_PID = NitrateOptimizer(dynamic_configuration=dynamic_config)
    simulation_PID.run()
    
    dynamic_config = set_dynamic_config()
    simulation_static_fert = NitrateOptimizer(dynamic_configuration=dynamic_config, static_fertigation=True)
    simulation_static_fert.run()
    

    plt.show() 