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








from phydrus_model_initializer import DynamicConfig, PhydrusModelInit
from static_configuration import static_config

class NitrateOptimizer():
    static_configuration = static_config
    current_step = 0
    max_step =200
    current_step_phydrus = None

    def __init__(self, static_configuration = static_config, dynamic_configuration = DynamicConfig()):
        static_configuration = static_configuration
        dynamic_configuration = dynamic_configuration
    def run(self):
        while self.current_step < self.max_step:
            self.step()
            current_step += 1

    def step(self):
        phydrus= PhydrusModelInit(self.dynamic_configuration, self.static_configuration)
        self.current_step_phydrus = phydrus
        self.update_init_params()
        self.update_reinforcement_learning_input_logs()
        self.compute_cost_function()
        self.change_percepitation()


    def update_init_params(self):
        """
        update the next steps initial params based on the previous steps output
        """
        node_info = self.phydrus.ml.read_nod_inf(times=[self.static_configuration["n_hours"]])
        self.static_configuration["initial_wc_distribution"] = node_info["Moisture"]
        self.static_configuration["initial_conc_distribution"] = node_info["Conc"]
        self.static_configuration["auto_wc_and_NO3"] = False

    def update_reinforcement_learning_input_logs(self):
        water_content = self.phydrus.get_node_info(column_name="theta")
        concentration = self.phydrus.get_node_info(column_name='Conc')
        # save in a format the reinforcement learning will be able to use efficiently

    def change_percepitation():
        """
        Here is the algorithm that updates the percipitation we would like to give
        """
        # recieve cost function for previous iteration
        # read reinforcement learning info
        # logic
        # dynamic_configuration["percipitation"] update, dynamic_configuration["fertigation_conc"] update