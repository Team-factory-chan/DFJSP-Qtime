# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 13:33:05 2022

@author: parkh
"""
    
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import copy
import random
from matplotlib import pyplot as plt
from Resource import *
from Job import *
from Event import *
from collections import defaultdict
from plotly.offline import plot
    
class FJSP_simulator(object):
        
    #processing time
    #setup time
    #queue time
    #realase time
    #duedate time(realase time에 포함)
    # 5가지 데이터를 모두 불러옴    
    def __init__(self, p_time_data, s_time_data,q_time_data, r_time_data ,k):
        # 고정 부분
        self.k = k #디스패칭룰 사용할 때 쓸 것
        
        self.process_time_table = pd.read_csv(p_time_data,index_col=(0))
        self.setup_time_table = pd.read_csv(s_time_data, index_col=(0))
        self.rtime_and_dtime_table = pd.read_csv(r_time_data, index_col=(0))
        self.queue_time_table = pd.read_csv(q_time_data, index_col=(0))
        
        self.machine_number = len(self.process_time_table.columns) #machine 개수
        
        """총 job 개수"""
        operation = self.process_time_table.index
        op_table=[]
        for i in range(len(operation)):
            op_table.append(operation[i][1:3])
        self.job_number = len(set(op_table)) #총 Job_type 개수
        
        """각 job type 별로 총 job 개수"""
        self.total_job = [0 for x in range(self.job_number)] # 생산 요구량 의미 [3,3,2,4,2,3] 
        """각 job별로 총 operation개수"""
        self.max_operation = [0 for x in range(self.job_number)]
        for i in range(1, self.job_number+1):
            for j in op_table:
                if i == int(j):
                    self.max_operation[i-1] +=1
        
        # 리셋 부분
        self.done = False #종료조건 판단
        self.remain_job = copy.deepcopy(self.total_job)
        #self.num_of_op = sum(self.total_operation)
        self.time = 0 #시간
        self.plotlydf = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.plotlydf_arrival_and_due = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.step_number = 0
        self.j = 0
        self.j2 = 0
        #Job 객체 , jop, machine객체, 현재시간, 끝나는 시간, 이벤트이름"track_in_finish", rule_name,step_num, setup_time
        
        """job 인스턴스 생성"""
        self.j_list = defaultdict(Job)
        self.event_list = []
        for i in range(len(self.rtime_and_dtime_table)):
            due_date = self.rtime_and_dtime_table.iloc[i]["d_time"]
            realase_date = self.rtime_and_dtime_table.iloc[i]["r_time"]
            if realase_date == 0:
                status = "WAIT"
            else:
                status = "NOTYET"
            
            job_type = self.rtime_and_dtime_table.iloc[i].name
            
            job_type_int = int(job_type[1:])
            job_id = job_type + "-" + str(realase_date) + "-" + str(i)    
            j = Job(job_id, job_type_int ,self.max_operation[job_type_int-1], self.setup_time_table[job_type]
                    ,due_date,self.queue_time_table.loc[job_type].tolist(), realase_date, "NOTYET") 
            self.j_list[j.id] = j
            
            e = Event(j,"job_arrival" , "NONE", self.time, realase_date,"job_arrival","NONE","NONE","NONE",0)
            start = datetime.fromtimestamp(self.time*3600)
            realase = datetime.fromtimestamp(realase_date*3600)
            due = datetime.fromtimestamp(due_date*3600)
            due_end = datetime.fromtimestamp((due_date+1)*3600)
            self.event_list.append(e)
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "job_arrival", JOB_ID = j.id  ,Task="job_arrival", Start=start, Finish=realase, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "job_arrival", Q_check = "job_arrival")
            self.j2 += 1
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "due_date", JOB_ID = j.id  ,Task="due_date", Start=due, Finish=due_end, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "due", Q_check = "due")
            self.j2 += 1
        """machine 인스턴스 생성"""
        self.r_list = defaultdict(Resource)
        for i in range(self.machine_number):
            r = Resource("M"+str(i+1))
            self.r_list[r.id] = r
            
        
        
        


    def reset(self):
        # 리셋 부분
        self.done = False #종료조건 판단
        self.remain_job = copy.deepcopy(self.total_job)
        #self.num_of_op = sum(self.total_operation)
        self.time = 0 #시간
        self.plotlydf = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.plotlydf_arrival_and_due = pd.DataFrame([],columns=['Type','JOB_ID','Task','Start','Finish','Resource','Rule','Step','Q_diff','Q_check'])
        self.step_number = 0
        self.j = 0
        self.j2 = 0
        #Job 객체 , jop, machine객체, 현재시간, 끝나는 시간, 이벤트이름"track_in_finish", rule_name,step_num, setup_time
        
        """job 인스턴스 생성"""
        self.j_list = defaultdict(Job)
        self.event_list = []
        for i in range(len(self.rtime_and_dtime_table)):
            due_date = self.rtime_and_dtime_table.iloc[i]["d_time"]
            realase_date = self.rtime_and_dtime_table.iloc[i]["r_time"]
            if realase_date == 0:
                status = "WAIT"
            else:
                status = "NOTYET"
            
            job_type = self.rtime_and_dtime_table.iloc[i].name
            
            job_type_int = int(job_type[1:])
            job_id = job_type + "-" + str(realase_date) + "-" + str(i)    
            j = Job(job_id, job_type_int ,self.max_operation[job_type_int-1], self.setup_time_table[job_type]
                    ,due_date,self.queue_time_table.loc[job_type].tolist(), realase_date, "NOTYET") 
            self.j_list[j.id] = j
            
            e = Event(j,"job_arrival" , "NONE", self.time, realase_date,"job_arrival","NONE","NONE","NONE",0)
            start = datetime.fromtimestamp(self.time*3600)
            realase = datetime.fromtimestamp(realase_date*3600)
            due = datetime.fromtimestamp(due_date*3600)
            due_end = datetime.fromtimestamp((due_date+1)*3600)
            self.event_list.append(e)
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "job_arrival", JOB_ID = j.id  ,Task="job_arrival", Start=start, Finish=realase, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "job_arrival", Q_check = "job_arrival")
            self.j2 += 1
            self.plotlydf_arrival_and_due.loc[self.j2] = dict(Type = "due_date", JOB_ID = j.id  ,Task="due_date", Start=due, Finish=due_end, Resource="NONE", Rule = "NONE", 
                                             Step = "NONE", Q_diff = "due", Q_check = "due")
            self.j2 += 1
        """machine 인스턴스 생성"""
        self.r_list = defaultdict(Resource)
        for i in range(self.machine_number):
            r = Resource("M"+str(i+1))
            self.r_list[r.id] = r
        
        s = [0] * 12
        
        df = pd.Series(s)
        s = df.to_numpy()
        
        return s
        
            
        return s
    
    def performance_measure(self):
        q_time_true = 0
        q_time_false = 0
        makespan = self.time
                    
        Flow_time = 0
        Tardiness_time = 0 #new
        Lateness_time = 0 #new
        T_max = 0 #new
        L_max = 0 #new
        value_time_table = []
        full_time_table = []
        machine_util = 0
        util = 0
        q_job_f = 0
        q_job_t = 0
        z = []
        total_q_time_over = 0
        makespan = self.time
        for machine in self.r_list:
            value_added_time, full_time = self.r_list[machine].util()
            value_time_table.append(value_added_time)
            full_time_table.append(full_time)
        util = sum(value_time_table)/sum(full_time_table)
        for job in self.j_list:
            Flow_time += self.j_list[job].job_flowtime
            if self.j_list[job].tardiness_time > T_max:
                T_max = self.j_list[job].tardiness_time
            Tardiness_time += self.j_list[job].tardiness_time
            Lateness_time += self.j_list[job].lateness_time
            k = []
            for q in self.j_list[job].q_time_check_list:
                k.append(q)
                if q > 0:
                    q_time_false += 1
                else:
                    q_time_true += 1
            z.append(k)
            if self.j_list[job].condition == True:
                q_job_t += 1
            else:
                q_job_f += 1
            total_q_time_over += self.j_list[job].cal_q_time_total()
        #fig = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        #fig.show()
        return Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max,q_time_true,q_time_false,q_job_t, q_job_f, total_q_time_over
    def modify_width(self, bar, width):
        """
        막대의 너비를 설정합니다.
        width = (단위 px)
        """
        bar.width = width
    def modify_text(self, bar):
        """
        막대의 텍스트를 설정합니다.
        width = (단위 px)
        """
        bar.text = "aasaas"
    def to_top_arrival_df(self, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """
        arrival_df = df.loc[df['Type'] == 'job_arrival']
        df = df[df['Type'] != 'job_arrival']
        arrival_df = arrival_df.append(df, ignore_index=True)
        return arrival_df
    
    def to_bottom_setup_df(self, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """ 
        setup_df = df.loc[df['Type'] == 'setup']
        df = df[df['Type'] != 'setup']
        df = df.append(setup_df, ignore_index=True)
        return df
    
    def to_bottom_due_df(self, df):
        """
        figure의 경우 위에서 부터 bar 생성됩니다.
        track_in event를 df(데이터프레임) 가장 밑 행으로 배치시킵니다.
        이 작업을 통해 TRACK_IN 이벤트가 다른 중복되는 차트에 가려지는 것을 방지합니다.
        """ 
        setup_df = df.loc[df['Type'] == 'due_date']
        df = df[df['Type'] != 'due_date']
        df = df.append(setup_df, ignore_index=True)
        return df
    
    def gannt_chart(self):
        
        step_rule = []
        for i in range(len(self.plotlydf)):
            if str(self.plotlydf["Rule"].loc[i])  != "None":
                step_rule.append(str(self.plotlydf["Step"].loc[i])+"-"+str(self.plotlydf["Rule"].loc[i]))
            else:
                step_rule.append("NONE")
        self.plotlydf["Step-Rule"] = step_rule
        
        id_op = []
        for i in range(len(self.plotlydf)):
            if str(self.plotlydf["Task"].loc[i])  != "None":
                id_op.append(str(self.plotlydf["JOB_ID"].loc[i])+"-"+str(self.plotlydf["Task"].loc[i]))
            else:
                id_op.append("NONE")
        self.plotlydf["ID_OP"] = id_op
        
        df = self.plotlydf
        
        fig = px.bar(df, x="Resource", y="Type", color="Type", facet_row="Type")
        fig.update_yaxes(matches=None)
        fig.show()
        
        
        
        plotlydf2 = self.plotlydf.sort_values(by=['Resource','Type'], ascending=False)
        df = self.to_bottom_setup_df(plotlydf2) #setup 뒤로 보낸 데이터 프레임
        
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", hover_data=['Rule'],template="simple_white",color="Type", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Task", width=2000, height=800)
        fig.update_traces(marker=dict(line_color="black"))
        
        [(self.modify_width(bar, 0.7))
        for bar in fig.data if ('setup' in bar.legendgroup)]
        #fig.show()
        
        #fig,write_html(f"{PathInfo.xlsx}{os.sep}temp_target.html", default_width=2300, default_height=900)
        plotlydf3 = self.plotlydf.sort_values(by=['Type'], ascending=True)
        fig2 = px.timeline(plotlydf3, x_start="Start", x_end="Finish", y="Type", template="seaborn" ,color="Resource",text = "Resource", width=2000, height=1000)
        fig2.update_traces(marker=dict(line_color="yellow", cmid = 1000))
        #fig2.show()
        
        fig3 = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", template="simple_white",color="Type", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Rule", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar)) for bar in fig3.data if ('setup' in bar.legendgroup)]
        #fig3.show()
        
        
        fig4 = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", template="simple_white",color="Type", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Step-Rule", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
        for bar in fig4.data if ('setup' in bar.legendgroup)]
        #fig4.show()
        
        fig5 = px.timeline(df, x_start="Start", x_end="Finish", y="Rule", template="simple_white",color="Rule", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Step-Rule", width=2000, height=800)
        fig5.show()
        
        fig6 = px.timeline(df, x_start="Start", x_end="Finish", y="Resource", template="simple_white",color="Type", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "ID_OP", width=2000, height=800)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
        for bar in fig6.data if ('setup' in bar.legendgroup)]
        
        df = self.plotlydf.sort_values(by=['Type'], ascending=True)
        
        fig7 = px.timeline(df, x_start="Start", x_end="Finish", y="JOB_ID", template="simple_white",color="Q_check", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Q_diff", width=2000, height=2000)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
        for bar in fig6.data if ('setup' in bar.legendgroup)]
        fig7.show()
        df = self.plotlydf_arrival_and_due.append(self.plotlydf, ignore_index=True)
        df = df.sort_values(by=['Start',"Finish"], ascending=[False, False])
        df = self.to_top_arrival_df(df)
        df = self.to_bottom_due_df(df)
        fig8 = px.timeline(df , x_start="Start", x_end="Finish", y="JOB_ID", template="simple_white",color="Q_check", color_discrete_sequence=px.colors.qualitative.Dark24 ,text = "Q_diff", width=2000, height=2000)
        [(self.modify_width(bar, 0.7), self.modify_text(bar))
        for bar in fig8.data if ('setup' in bar.legendgroup)]
        #fig8.show()
        plot(fig8)
        print(df)
    #오퍼레이션 길이 50,메이크스팬 100, Max op 5,Min op 5, Max-min
    #39025431
    
    def step(self, action):
       # print(self.num_of_op)
        #print(self.total_job)
        ##print(self.total_operation)
        done = False
        while True:
            machine = self.check_availability()
            if machine == "NONE":
                self.process_event()
                #이벤트도 비워져 있고, #job들도 다 done이면 종료
                if len(self.event_list) == 0 and all(self.j_list[job].status == "DONE" for job in self.j_list): 
                    done = True
                    s_prime = self.set_state()
                    r =  0
                    break
            else:
                p_time,jop = self.dispatching_rule_decision(machine, action)
                s_prime = self.set_state()
                reservation_time = self.r_list[machine].reservation_time
                last_work_finish_time = self.r_list[machine].last_work_finish_time
                max_reservation = 0
                min_reservation = 100000000
                p_time_lst = []
                total_idle = 0
                for machine in self.r_list:
                    p_time_lst.append(self.process_time_table[machine].loc[jop])
                    if self.r_list[machine].reservation_time > max_reservation:
                        max_reservation = self.r_list[machine].reservation_time
                    if self.r_list[machine].reservation_time < min_reservation:
                        min_reservation = self.r_list[machine].reservation_time
                    if self.r_list[machine].reservation_time < last_work_finish_time:
                        total_idle += (last_work_finish_time - self.r_list[machine].reservation_time)
                        self.r_list[machine].reservation_time = last_work_finish_time
                r = -(reservation_time-last_work_finish_time + total_idle)
                break
        return s_prime, r , done
    
    def set_state(self):
        """
        재공 정보 :
            대기 중인 job들의 개수
            작업 중인 job들의 개수
            대기 중인 job들의 남은 operation 개수 평균
            대기 중인 job들의 tardiness 평균
            대기 중인 job들의 q-time 초과 평균
            대기 중인 job들의 flow time 평균
        
        기계 정보 :
            기계의 현재 시간
            현재 시간 / 다른 기계의 최대 시간
            다른 기계들과 차이의 평균
        
        누적 정보 :
            현재까지 total tardiness
            현재까지 total q over time
            현재까지 처리한 job 개수
        """
        s = []
        number_of_jobs_wait = 0 #clear
        number_of_jobs_load = 0 #clear
        total_remain_operation = 0
        total_tardiness = 0
        total_q_time_over = 0
        total_flow_time = 0
        number_of_job_done = 0 #clear
        
        total_job_tardiness_done =0 #clear
        total_job_q_time_over_done = 0 # clear
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                number_of_jobs_wait += 1
                total_remain_operation += self.j_list[job].remain_operation
                total_tardiness += self.j_list[job].cal_tardiness(self.time)
                total_q_time_over += self.j_list[job].cal_q_time(self.time)
                total_flow_time += self.j_list[job].cal_flowtime(self.time)
            elif self.j_list[job].status == "PROCESSING":
                number_of_jobs_load += 1
            elif self.j_list[job].status == "DONE":
                number_of_job_done += 1
                total_job_tardiness_done += self.j_list[job].tardiness_time
                q_total = self.j_list[job].cal_q_time_total()
                total_job_q_time_over_done += q_total
        s.append(number_of_jobs_wait)
        s.append(number_of_jobs_load)
        if number_of_jobs_wait == 0:
            s.append(0)
            s.append(0)
            s.append(0)
            s.append(0)
        else:
            s.append(total_remain_operation / number_of_jobs_wait)
            s.append(total_tardiness / number_of_jobs_wait)
            s.append(total_q_time_over / number_of_jobs_wait)
            s.append(total_flow_time / number_of_jobs_wait)
    
        current_time = self.time
        total_reservation_time_diff = 0
        max_reservation_time = 0
        for machine in self.r_list:
            total_reservation_time_diff += self.r_list[machine].reservation_time - current_time
            if max_reservation_time > self.r_list[machine].reservation_time:
                max_reservation_time = self.r_list[machine].reservation_time
        
        s.append(current_time)
        if max_reservation_time == 0 :
            s.append(0)
        else:
            s.append(current_time / max_reservation_time)
        s.append(total_reservation_time_diff / len(self.r_list))
        
        s.append(number_of_job_done)
        if number_of_job_done == 0:
            s.append(0)
            s.append(0)
        else:
            s.append(total_job_tardiness_done / number_of_job_done)
            s.append(total_job_q_time_over_done / number_of_job_done)
        
        df = pd.Series(s)
        s = df.to_numpy()
        
        return s
    def run(self):
        while True:
            machine = self.check_availability()
            if machine != "NONE":
                p_time = self.dispatching_rule_decision(machine, self.k)
            else:
                if len(self.event_list) == 0:
                    break
                self.process_event()
                
        
        Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time = self.performance_measure()
        print(self.k)
        print("FlowTime:" , Flow_time)
        print("machine_util:" , machine_util)
        print("util:" , util)
        print("makespan:" , makespan)
        print("Tardiness:" , tardiness)
        print("Lateness:" , lateness)
        print("T_max:" , t_max)
        print("Q time True", q_time_true)
        print("Q time False", q_time_false)
        print("Q job True", q_job_t)
        print("Q job False", q_job_f)
        print("Q total over time", q_time)
        #self.gannt_chart()
        return Flow_time, util, makespan
    #event = (job_type, operation, machine_type, start_time, end_time, event_type)
    def dispatching_rule_decision(self,machine, a):
        #print(machine)
        if a == "random":
            coin = random.randint(0,1)
        else:
            coin = int(a)
        if coin == 0:
            p_time,jop = self.dispatching_rule_SPT(machine)
        elif coin == 1:
            p_time,jop = self.dispatching_rule_SSU(machine)
        elif coin == 2:
            p_time,jop = self.dispatching_rule_SPTSSU(machine)
        elif coin == 3:
            p_time,jop = self.dispatching_rule_MOR(machine)   
        elif coin == 4:
            p_time,jop = self.dispatching_rule_LOR(machine)
        elif coin == 5:
            p_time,jop = self.dispatching_rule_EDD(machine)
        elif coin == 6:
            p_time,jop = self.dispatching_rule_MST(machine)
        elif coin == 7:
            p_time,jop = self.dispatching_rule_FIFO(machine)
        elif coin == 8:
            p_time,jop = self.dispatching_rule_LIFO(machine)
        elif coin == 9:
            p_time,jop = self.dispatching_rule_CR(machine)
        
        return p_time,jop
    def process_event(self):
        #print(self.event_list)
        self.event_list.sort(key = lambda x:x.end_time, reverse = False)
        event = self.event_list.pop(0)
        self.time = event.end_time
        if event.event_type == "job_arrival":
            event.job.arrival()
        else:
            if event.event_type == "setup_change":
                event_type = "setup"
            else:
                event_type = "j"+str(event.job.job_type)
                last = event.job.complete_setting(event.start_time, event.end_time ,event.event_type) # 작업이 대기로 변함, 시작시간, 종료시간, event_type
                event.machine.complete_setting(event.start_time, event.end_time ,event.event_type) # 기계도 사용가능하도록 변함
            rule = event.rule_name
            step = event.step_num
            start = datetime.fromtimestamp(event.start_time*3600)
            end = datetime.fromtimestamp(event.end_time*3600)
            q_time_diff = event.q_time_diff
            q_time_check = event.q_time_check
            #print(self.step_number) Q_Check , Q_time_over
            self.plotlydf.loc[self.j] = dict(Type = event_type, JOB_ID = event.job.id  ,Task=event.jop, Start=start, Finish=end, Resource=event.machine.id, Rule = rule, 
                                             Step = step, Q_diff = q_time_diff, Q_check = q_time_check) #간트차트를 위한 딕셔너리 생성, 데이터프레임에 집어넣음
            self.j+=1
    
    def assign_setting(self, job, machine,reservation_time): #job = 1 machine = 1
        q_time_diff = job.assign_setting(machine, self.time)
        if job.remain_operation == 0:
            self.total_job[job.job_type-1] -= 1
        #self.total_operation[job.job_type-1] -=1
        machine.assign_setting(job, reservation_time)
        return q_time_diff
        
    def check_availability(self):
        index_k = 0
        select_machine = "NONE"
        for machine in self.r_list:
            index_k += 1
            if self.r_list[machine].status == 0:
                machine = self.r_list[machine].id #machine 이름
                p_table=[]
                for job in self.j_list: #job 이름과 operation이름 찾기
                    jop = self.j_list[job].jop()
                    if jop not in self.process_time_table.index: #해당 jop가 없는 경우  
                        pass
                    elif self.process_time_table[machine].loc[jop] == 0 : #해당 jop가 작업이 불가능할 경우
                        pass
                    elif self.j_list[job].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    else:
                        p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop]])
                if len(p_table) == 0:#현재 이벤트를 발생시킬 수 없음
                    pass
                else:
                    select_machine = machine
                    break
        return select_machine
                
    def dispatching_rule_SPT(self, machine):
        rule_name= "SPT"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop])
        
        p_table.sort(key = lambda x:x[1], reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_SSU(self, machine):
        rule_name= "SSU"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x:x[3], reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time,q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_SPTSSU(self, machine):
        rule_name= "SPTSSU"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: x[1] + x[3], reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_MOR(self, machine):
        rule_name= "MOR"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: x[0].remain_operation, reverse = True)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_LOR(self, machine):
        rule_name= "LOR"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: x[0].remain_operation, reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_EDD(self, machine):
        rule_name= "EDD"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: x[0].duedate, reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_MST(self, machine):
        rule_name= "MST"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: x[0].duedate - self.time - x[1], reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_CR(self, machine):
        rule_name= "CR"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: (x[0].duedate - self.time) / x[1], reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_FIFO(self, machine):
        rule_name= "FIFO"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: x[0].job_arrival_time, reverse = False)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    
    def dispatching_rule_LIFO(self, machine):
        rule_name= "LIFO"
        step_num = self.step_number
        self.step_number+=1
        machine = self.r_list[machine].id #machine 이름
        p_table=[]
        for job in self.j_list: #job 이름과 operation이름 찾기
            if self.j_list[job].status == "WAIT":
                jop = self.j_list[job].jop()
                setup_time = self.j_list[job].setup_table['j'+str(self.r_list[machine].setup_status)]
                if self.process_time_table[machine].loc[jop] != 0:
                    p_table.append([self.j_list[job], self.process_time_table[machine].loc[jop],jop,setup_time])
        
        p_table.sort(key = lambda x: x[0].job_arrival_time, reverse = True)
        setup_time = p_table[0][0].setup_table['j'+str(self.r_list[machine].setup_status)] #컬럼에서 machine에 세팅되어있던 job에서 변경유무 확인
        jop = p_table[0][2]
        if setup_time !=0:
            e = Event(p_table[0][0],"setup" , self.r_list[machine], self.time, self.time+setup_time,"setup_change","NONE",step_num,setup_time, 0)
            self.event_list.append(e)
        q_time_diff = self.assign_setting(p_table[0][0], self.r_list[machine],self.time+setup_time+p_table[0][1])
        e = Event(p_table[0][0], jop ,self.r_list[machine], self.time, self.time+setup_time+p_table[0][1],"track_in_finish",rule_name,step_num,setup_time, q_time_diff)
        self.event_list.append(e)
        return p_table[0][1], jop
    




makespan_table = []
util = []
ft_table = []

for i in range(2,3):
    main = FJSP_simulator('/DFJSP_test.csv','C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_setup_test.csv',
                          "C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_Qdata_test.csv","C:/Users/parkh/git_tlsgudcks/simulator/data/DFJSP_rdData_test2.csv",i)
    FT, util2, ms = main.run()
    makespan_table.append(ms)
    util.append(util2)
    ft_table.append(FT)
print(makespan_table)
print(ft_table)
print(util)
