# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:19:17 2023

@author: parkh
"""
import pandas as pd

#todo Job이란 이름 Lot로 변경
class Lot(object):

    # Default Constructor
    def __init__(self, lot_id, job_id, job_type, max_operation, duedate, arrival_time, status, oper_list, q_time_table):
        # 고정 정보
        self.id = lot_id # job번호
        self.job_id = job_id
        self.job_type = job_type # job type이 뭔지
        self.max_operation = max_operation # 이 job의 max operation이 언젠지
        self.duedate = duedate # 이 job의 duedate가 언제인지
        self.oper_list = oper_list
        self.q_time_table = q_time_table
        """
        공정 수 만큼 존재함
        [10, 14, 25, 34] J11 -> J12를 작업할 때 까지 10시간 안에 작업해야함
        """
        self.job_arrival_time = arrival_time
        
        #변화하는 데이터
        self.oper_number = 0
        self.current_operation_id = oper_list[self.oper_number]  # 현재 공정이 어디인지
        #status 종류 -> "WAIT", "NOTYET", "DONE", "PROCESSING"
        self.status = status # 현재 job의 상태가 무엇인지
        self.remain_operation = self.max_operation #이 job의 남은 operation이 몇 개인지
        self.start_time = 0
        self.condition = True
        self.act_end_time = 0
        # For History and KPI
        
        self.history_list = []
        self.job_flowtime = 0 # job의 flow time
        self.tardiness_time = 0 #job의 tardiness time
        self.lateness_time = 0 # job의 lateness time
        self.operation_in_machine = [0 for x in range(max_operation)] #각각의 operation이 어떤 machine에 있었는지
        self.q_time_check_list = {}
    
    def assign_setting(self, machine, assign_time):
        machine_id = machine.id
        #todo 각각의 공정이 어떤 machine에서 작업됐는지 기록
        self.status = "PROCESSING"
        self.remain_operation -= 1
        q_time_diff = "None"
        if self.oper_number != 0:
            if self.q_time_table[self.current_operation_id] == 0:
                q_time_diff = 0
            else:
                q_time_diff = max(0, (assign_time - self.start_time) - self.q_time_table[self.current_operation_id])
            self.q_time_check_list[self.current_operation_id] = q_time_diff
            if q_time_diff > 0:
                self.condition = False

        self.oper_number += 1
        if self.oper_number == self.max_operation:
            self.current_operation_id = "DONE"
        else:
            self.current_operation_id = self.oper_list[self.oper_number]
        return q_time_diff

    def oper_check_for_meta(self):
        self.oper_number += 1
        if self.oper_number == self.max_operation:
            self.oper_number = 0
            self.current_operation_id = self.oper_list[self.oper_number]
        else:
            self.current_operation_id = self.oper_list[self.oper_number]

    def complete_setting(self,start_time, end_time, event_type):
        self.status = "WAIT"
        last = False
        self.start_time = end_time #q_time을 체크하는 time임
        if event_type == "track_in_finish" and self.remain_operation == 0:
            self.job_flowtime += end_time - self.job_arrival_time
            self.tardiness_time = max(0 , end_time-self.duedate)
            self.lateness_time = end_time-self.duedate
            last = True
        if last == True:
            self.status = "DONE"
        self.act_end_time = end_time
        return last

    def arrival(self):
        self.status = "WAIT"

    def change_next_oper(self):
        self.oper_number += 1
        
    def cal_flowtime(self, c_time):
        flow = c_time - self.job_arrival_time
        return flow
    
    def cal_tardiness(self, c_time):
        tardiness = max(0, c_time - self.duedate)
        return tardiness
    
    def cal_q_time(self, c_time):
        if self.start_time == 0:
            return 0
        else:
            q_time_diff = max(0, (c_time - self.start_time) - self.q_time_table[self.current_operation_id])
            return q_time_diff
        
    def cal_q_time_total(self):
        total_q = sum(self.q_time_check_list.values())
        return total_q

    def check_q_time(self, c_time):
        q_time_diff = self.cal_q_time(c_time)
        if q_time_diff == 0 and self.start_time == 0:
            return 2
        elif q_time_diff == 0:
            return 1
        else:
            return 0














