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



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from phydrus_model_initializer import DynamicConfig, PhydrusModelInit
from static_configuration import static_config

class NitrateOptimizer():
    past_days_data_file = "water_content_and_fertigation_history.csv"
    static_configuration = None
    dynamic_configuration = None
    current_step = 0
    max_step = 80
    phydrus = None
    irigation_fertigation_log = []

    def __init__(self, static_configuration = static_config, dynamic_configuration = DynamicConfig()):
        self.static_configuration = static_configuration
        self.dynamic_configuration = dynamic_configuration

    def clear_past_days_data(self):
        # Read the header of the CSV file without loading the data
        df = pd.read_csv(self.past_days_data_file)

        # Save only the headers back to the file (overwrite the file with only headers)
        df.head(0).to_csv(self.past_days_data_file, index=False)  # head(0) returns only the header


    def run(self):
        self.clear_past_days_data()
        while self.current_step < self.max_step:
            self.step()
            self.current_step += 1
        self.plot_nitrate_concentration()
        print(self.irigation_fertigation_log)

    def step(self):
        phydrus= PhydrusModelInit(self.dynamic_configuration, self.static_configuration)
        self.phydrus = phydrus
        self.update_init_params()
        self.update_past_data_file()
        self.compute_cost_function()
        irrigation_fertigation = self.decide_irrigation_and_fertigation()
        self.irigation_fertigation_log.append(irrigation_fertigation)

    def compute_cost_function(self):
        return 0
    
    def update_init_params(self):
        """
        update the next steps initial params based on the previous steps output
        """
        node_info = self.phydrus.ml.read_nod_inf(times=[self.static_configuration["n_hours"]])
        self.static_configuration["initial_wc_distribution"] = node_info["Moisture"]
        self.static_configuration["initial_conc_distribution"] = node_info["Conc(1..NS)"]
        self.static_configuration["auto_wc_and_NO3"] = False

    def update_past_data_file(self):
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

    def plot_nitrate_concentration(self, goal_column="Conc_20", goal_value=40):
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
        """
        Decide how much precipitation and fertigation to apply
        based on past water content and nitrate concentration.
        """
        past_data = self.load_past_data(1)  # Use last 3 days for decision-making

        if past_data is None or past_data.empty:
            print("No past data available, using default irrigation.")
            return 0.5, 40  # Default values for precipitation (mm) and fertigation (kg/ha)

        # Example logic:
        avg_wc_20cm = past_data["WC_20"].mean()
        avg_nitrate_20cm = past_data["Conc_20"].mean()

        # Adjust irrigation based on water content at 30cm depth
        if avg_wc_20cm < 0.2:  # Too dry
            precipitation = 2
        elif avg_wc_20cm > 0.32:  # Too wet
            precipitation = 0.2
        else:
            precipitation = 0.8  # Normal irrigation

        # Adjust fertigation based on nitrate concentration
        if avg_nitrate_20cm < 30:
            fertigation = 150  # Increase nitrogen
        elif avg_nitrate_20cm > 70:
            fertigation = 10  # Reduce nitrogen
        else:
            fertigation = 80 # Normal fertigation

        self.dynamic_configuration["precipitation"] = precipitation
        self.dynamic_configuration["fertigation_conc"] = fertigation
        return precipitation, fertigation


if __name__ == "__main__":
    simulation = NitrateOptimizer()
    simulation.run()