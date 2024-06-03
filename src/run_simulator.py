from common.Parameters import *
import yaml
from simlator.Simulator import *
class Run_Simulator:
    def __init__(self):
        print("simulator on")
        Parameters.set_time_to_string() # 현재 시간 가져오는 코드 -> 로그 및 기록을 위함
        Parameters.set_absolute_path()

        with open(f'{pathConfig.absolute_path}/hyperparameter.yaml', 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        Parameters.init_parameter_setting(config_data['engine'])
        Parameters.init_db_setting(config_data['database'])
        Parameters.set_dataSetId(["sks_train_1"])  # 사용할 데이터셋 설정
        Parameters.set_plan_horizon(840)

        Simulator.init_simulator(Parameters.datasetId) # 데이터셋 선택 후 데이터에 맞게 시뮬레이터 설정
        print("set complete")
    def main(self, mode, dsp_rule):
        logging.info(f"mode: {mode}")
        logging.info(f"dsp_rule: {dsp_rule}")
        if mode == "DSP_run":
            Simulator.run(dsp_rule)

if True:
    simulator = Run_Simulator()
    # 사용 가능 DSP rule = SPT, SPTSSU, SSU, LOR, MOR, EDD , MST, FIFO, LIFO
    simulator.main("DSP_run","SPTSSU") # dsp_rule = 개별 확인할 때만 사용하면 됨

# gantt chart 쑬 것인지
# 학습 방법, kpi목표
# 모든 디스패칭 룰 돌리기
