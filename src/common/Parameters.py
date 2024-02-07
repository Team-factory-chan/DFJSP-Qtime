
import datetime
import logging

#todo 파라미터 정리 필요, 강화학습 -> 하이퍼 파라미터
class Parameters:
    """
    "p_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_test.csv",
    "s_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_setup_test.csv",
    "q_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_Qdata_test.csv",
    "rd_data" : "/Users/shin/DFJSP-Qtime/Data/DFJSP_rdData_test2.csv"
    """
    # 여기에 파라미터를 초기화합니다.'
    print("parameter load")

    r_param = {
        # 강화학습 파라미터
        "gamma": 0.99,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "buffer_limit": 50000,
        "input_layer" : 51,
        "output_layer" : 10,
        "episode" : 2000
    }

    db_setting =  {
        "host" : 'localhost',
        "port" : 3306,
        "user" : 'root',
        "passwd" : '1234',
        "db" : 'fjsp_simulator_db',
        "charset" : 'utf8'
    }

    DSP_rule_check ={
        #DSP_run에서 사용할 check
        "SPT": True,
        "SSU": True,
        "SPTSSU": True,
        "MOR": True,
        "LOR": True,
        "EDD": True,
        "MST": True,
        "FIFO": True,
        "LIFO": True,
        "CR": True,
        "SQT" : True

    }


    gantt_on = {
        "mahicne_on_job_number" : False,
        "machine_gantt" : False,
        "DSP_gantt" : True,
        "step_DSP_gantt" : False,
        "heatMap_gantt" : False,
        "main_gantt" : True,
        "job_gantt_for_Q_time" : False
    }

    gantt_on_check = True # 간트 껏다 키기
    log_on = True #common 껐다 키기
    param_down_on = True #파라미터 다운 끄기
    meta_ver = True
    log_history = True

    db_data = ""

    reward_type = ""
    state_type = ""
    action_type = ""

    action_count = ""
    action_dimension = ""

    do_nothing_time = 24
    plan_horizon = 100000000
    simulation_time = ""
    log_path = ""

    save_log_directory = "/Users/shin/DFJSP-Qtime//log_data/"
    save_parameter_directory = "/Users/shin/DFJSP-Qtime//params_data/"
    #/Users/shin/DFJSP-Qtime/params_data
    @classmethod
    def set_check_gantt_on(cls, check_gantt):
        cls.gantt_on_check = check_gantt

    @classmethod
    def set_check_log(cls, check_log):
        cls.log_on = check_log

    @classmethod
    def set_time_to_string(cls):
        current_time = datetime.datetime.now()
        # 원하는 문자열 형식으로 시간을 포맷팅
        time_format = "%y%m%d_%H%M%S"  # "년월일_시분" 형태의 포맷
        current_time_str = current_time.strftime(time_format)
        cls.simulation_time = current_time_str

    @classmethod
    def set_check_down_parameter(cls, check_param):
        cls.param_down_on = check_param

    @classmethod
    def set_meta_ver(cls, meta_ver_check):
        cls.meta_ver = meta_ver_check

    @classmethod
    def set_reward_type(cls, reward_type):
        cls.reward_type = reward_type

    @classmethod
    def set_check_history_db(cls, check):
        cls.log_hitory = check

    @classmethod
    def set_log_path(cls):
        if cls.log_on == True:
            log_path = cls.save_log_directory + cls.simulation_time + "performance.common"
            logging.basicConfig(
                filename=log_path,
                level=logging.INFO,  # 로그 레벨을 INFO로 설정
                format='%(asctime)s [%(levelname)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

    @classmethod
    def set_dataSetId(cls, dataSetId):
        cls.db_data = dataSetId

    @classmethod
    def set_state_dimension(cls , Simulator):
        if cls.state_type == "state_12":
            cls.r_param['input_layer'] = 12
        elif cls.state_type == 'default_state':
            cls.r_param['input_layer'] = 8
        elif cls.state_type == "state_36":
            cls.r_param['input_layer'] = Simulator.number_of_machine * 3 + 6
        elif cls.state_type == 'cnn_state':
            cls.r_param['input_layer'] = 29
    @classmethod
    def set_state_type(cls, state):
        cls.state_type = state

    @classmethod
    def set_plan_horizon(cls, horizon):
        cls.plan_horizon = horizon






