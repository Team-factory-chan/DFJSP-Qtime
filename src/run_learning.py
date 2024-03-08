from learner.Algorithm.PPO import PPO
from src.learner.Algorithm.DQN import *
from src.learner.Algorithm.DQN_action_masking import *
from src.learner.Algorithm.DQN_CNN import *
from src.learner.Algorithm.DDQN import *
from src.learner.common.Hyperparameters import *
import yaml

class Run_Simulator:
    def __init__(self):
        Parameters.set_time_to_string()  # 현재 시간 가져오는 코드 -> 로그 및 기록을 위함
        Parameters.set_absolute_path()

        Parameters.set_dataSetId(["sks_train_1"])  # 사용할 데이터셋 설정
        #Parameters.set_dataSetId(['sks_train_11'])

        with open(f'{pathConfig.absolute_path}{os.sep}hyperparameter.yaml', 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        Parameters.init_parameter_setting(config_data['engine'])
        Parameters.init_db_setting(config_data['database'])
        Parameters.set_plan_horizon(840)
        action_list = ["SPTSSU", "SSU", "EDD", "MST", "FIFO", "LIFO"]

        Simulator._init(Parameters.datasetId)  # 데이터셋 선택 후 데이터에 맞게 시뮬레이터 설정

        Hyperparameters.init_hyperparameter_setting(config_data['hyperparameter'])
        Hyperparameters.init_rl_config_setting(config_data['configRL'], action_list,Simulator)


        print("set complete")


    def main(self, mode, algorithm):
        logging.info(f"mode: {mode}")
        logging.info(f"dsp_rule: {algorithm}")
        if mode == "learning":
            if algorithm == 'dqn':
                if ActionManager.action_type == "action_masking":
                    DQN_Action_Masking.main()
                else:
                    DQN.main()
            elif algorithm == 'ddqn':
                DDQN.main()
            elif algorithm == 'dqn_cnn':
                DQN_CNN.main()
            elif algorithm == 'PPO':
                ppo =PPO()
                ppo.main()
        elif mode == 'evaluate':
            if algorithm == "dqn":
                DQN.get_evaluate(f"{pathConfig.model_save_path}{os.sep}240209_233447", 100,
                                 ["sks_train_1"])
        elif mode == "result":
            if algorithm == 'dqn':
                DQN.get_result(f"{pathConfig.model_save_path}{os.sep}240209_233447{os.sep}24param.pt", ["sks_train_1"])

if True:
    simulator = Run_Simulator()
    simulator.main("learning","PPO") # dsp_rule = 개별 확인할 때만 사용하면 됨

# gantt chart 쑬 것인지
# 학습 방법, kpi목표
# 모든 디스패칭 룰 돌리기
