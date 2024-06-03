# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:15:22 2023

@author: parkh
"""

class Resource(object):
    # Default Constructor
    def __init__(self, resource_id, machine_type, setup_time_table, factory_id):
        self.id = resource_id #기계 아이디
        self.machine_type = machine_type # 기계 타입
        self.factory_id = factory_id
        #TODO wait와 processing으로 변경
        self.status = "WAIT" #기계 작동유무 , 0은 쉬는 중, 1은 작동중
        self.setup_status = "NONE" #기계 셋업 상태
        self.last_work_finish_time = 0 #최근에 끝난 작업시간
        self.job_id_processed = 0 #작업되고 있는 job ID
        self.utilization = 0
        self.idle_time = 0
        self.value_added_time = 0
        self.reservation_time = 0
        self.setup_time_table = setup_time_table #{j01 : {"j02" : 5, ...} , ...}
        # For History
        self.history_list = []

    def assign_setting(self, job, reservation_time):
        self.status = "PROCESSING"
        self.setup_status = job.job_type
        self.job_id_processed = job.job_type
        self.reservation_time = reservation_time
        
    def complete_setting(self,start_time, end_time, event_type):
        #todo 작업이 완료되고 나서야 이전 idletime이 계산됨 , 상관없을지도..
        if self.last_work_finish_time != start_time:
            self.idle_time += start_time - self.last_work_finish_time #setup이거나 idel이거나
        if event_type == "track_in_finish":
            self.status = "WAIT"
            self.job_id_processed = "NONE"
            self.value_added_time += end_time-start_time
            self.last_work_finish_time = end_time
        elif event_type == "setup_change":
            self.idle_time += end_time-start_time
        elif event_type == "NOTHING": #아무것도 하지 않는 액션을 취했을경우 -> 이건 아직 개발 x
            self.idle_time += end_time-start_time
        #self.last_work_finish_time = end_time
            
    def cal_util(self):
        return self.value_added_time, self.idle_time + self.value_added_time

    def cal_util2(self):
        return (self.value_added_time / (self.idle_time + self.value_added_time)) if (self.idle_time + self.value_added_time) > 0 else 0
    # Clone constructor 대신 이 함수를 사용

    def get_setup_time(self, toJobType):
        if self.setup_status != "NONE":
            setup_time = self.setup_time_table[self.setup_status][toJobType]
        else:
            setup_time = 0
        return setup_time

    def do_nothing_setting(self):
        self.status = "DoNothing"

    def wake_up_setting(self):
        self.status = "WAIT"

