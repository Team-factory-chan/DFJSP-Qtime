from src.common.Parameters import *
from src.master_db.DataInventory import *
class Hyperparameters:
    gamma = 0.99
    learning_rate = 0.0001
    batch_size = 32
    buffer_limit = 50000
    input_layer =  51
    output_layer =10
    episode =  2000

    reward_type = ""
    state_type = ""
    action_type = ""

    action_count = ""
    action_dimension = ""

    @classmethod
    def init_hyperparameter_setting(cls,config):
        cls.gamma = config['gamma']
        cls.learning_rate = config['learning_rate']
        cls.batch_size = config['batch_size']
        cls.buffer_limit = config['buffer_limit']
        cls.input_layer = config['input_layer']
        cls.output_layer = config['output_layer']
        cls.episode = config['episode']

    @classmethod
    def init_rl_config_setting(cls, config, action_list, simulator):
        cls.state_type = config['state_type']
        cls.reward_type = config['reward_type']
        cls.action_type = config['action_type']
        cls.set_action_type(cls.action_type, action_list)
        cls.set_state_dimension(simulator)

    @classmethod
    def set_action_type(cls, action_type, action_list):
        if action_type == "dsp_rule":
            cls.action_type = action_type
            cls.action_count = len(action_list)
            cls.action_dimension = [rule for rule in Parameters.DSP_rule_check.keys() if rule in action_list]
        elif action_type == "setup" or action_type == "action_masking":
            cls.action_type = action_type
            #dataset Id 여러 개 처리가능
            DataInventory.set_db_data(Parameters.datasetId[0])
            cls.action_dimension = [job.jobType for job in DataInventory.master_data["Job_db"]]
            cls.action_count = len(Hyperparameters.action_dimension)
        cls.output_layer = cls.action_count
    @classmethod
    def set_reward_type(cls, reward_type):
        cls.reward_type = reward_type

    @classmethod
    def set_state_type(cls, state):
        cls.state_type = state

    @classmethod
    def set_state_dimension(cls, Simulator):
        if cls.state_type == "state_12":
            cls.input_layer = 12
        elif cls.state_type == 'default_state':
            cls.input_layer = 8
        elif cls.state_type == "state_36":
            cls.input_layer = Simulator.number_of_machine * 3 + 6 -2
        elif cls.state_type == 'cnn_state':
            cls.input_layer = 29