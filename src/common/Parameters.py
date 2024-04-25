
import datetime
import logging
from src.common.pathConfig import *

#todo 파라미터 정리 필요, 강화학습 -> 하이퍼 파라미터
class Parameters:
    gantt_on_check = False  # 간트생성 on/off
    log_on = False  # log 기록 on/off
    param_down_on = False  # 모델 저장 on/off
    meta_ver = True     # metaheuristic 버전 on/off
    log_history = False  # lot 이력 db로 전송 여부

    datasetId = ""              # datasetId
    do_nothing_time = 24        # donothing time
    plan_horizon = 100000000    # 계획 구동 기간
    simulation_time = ""        # 학습 시작 시간

    # 여기에 파라미터를 초기화합니다.
    # db 커넥션 관련 파라미터 입니다
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
        "SQT" : True,
        "LPT" : True

    }

    gantt_on = {
        "mahicne_on_job_number" : False,
        "machine_gantt" : False,
        "DSP_gantt" : False,
        "step_DSP_gantt" : False,
        "heatMap_gantt" : False,
        "main_gantt" : True,
        "job_gantt_for_Q_time" : False
    }

    @classmethod
    def init_parameter_setting(cls, config):
        print("setting")
        cls.set_check_log(config["log_on"])
        cls.set_check_gantt_on(config['gantt_on_check'])
        cls.set_check_down_parameter(config['param_down_on'])
        cls.set_meta_ver(config['meta_ver'])
        cls.set_check_history_db(config['log_history'])
        cls.do_nothing_time = config['do_nothing_time']
        cls.set_plan_horizon(config['plan_horizon'])

    @classmethod
    def init_db_setting(cls, config):

        cls.db_setting['host'] = config['host']
        cls.db_setting['port'] = config['port']
        cls.db_setting['user'] = config['user']
        cls.db_setting['passwd'] = config['passwd']
        cls.db_setting['db'] = config['db']
        cls.db_setting['charset'] = config['charset']


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
    def set_check_history_db(cls, check):
        cls.log_hitory = check

    @classmethod
    def set_dataSetId(cls, dataSetId):
        cls.datasetId = dataSetId

    @classmethod
    def set_plan_horizon(cls, horizon):
        cls.plan_horizon = horizon

    @classmethod
    def set_absolute_path(cls):
        pathConfig.set_absolute_path()






