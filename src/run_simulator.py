from common.Parameters import *
from simlator.Simulator import *
class Run_Simulator:
    def __init__(self):
        print("simulator on")
        Parameters.set_dataSetId(["sks_train_1"])  # 사용할 데이터셋 설정
        Parameters.set_time_to_string() # 현재 시간 가져오는 코드 -> 로그 및 기록을 위함
        Parameters.set_check_log(True)  # common 남기기 여부
        Parameters.set_check_history_db(False)
        Parameters.set_check_gantt_on(True)  # 간트차트 띄우기 여부
        Parameters.set_meta_ver(False)
        Parameters.set_plan_horizon(840)

        Parameters.set_log_path() # 저장 경로 설정하는 코드
        Simulator.init_simulator(Parameters.db_data) # 데이터셋 선택 후 데이터에 맞게 시뮬레이터 설정
        print("set complete")
    def main(self, mode, dsp_rule):
        logging.info(f"mode: {mode}")
        logging.info(f"dsp_rule: {dsp_rule}")
        if mode == "DSP_run":
            Simulator.run(dsp_rule)
            #self.simulator.run(dsp_rule)
        elif mode == "DSP_check_run":
            for i in Parameters.DSP_rule_check:
                if Parameters.DSP_rule_check[i]:
                    print(i)
                    Simulator.run(i)
                    Simulator.reset(Parameters.db_data)

if True:
    simulator = Run_Simulator()
    simulator.main("DSP_run","SSU") # dsp_rule = 개별 확인할 때만 사용하면 됨

# gantt chart 쑬 것인지
# 학습 방법, kpi목표
# 모든 디스패칭 룰 돌리기
