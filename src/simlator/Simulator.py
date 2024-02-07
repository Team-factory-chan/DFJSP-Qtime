from src.common import Parameters
from src.Object.Event import *
from collections import defaultdict
from src.Object.Resource import *

from src.learner.Manager.StateManager import *
from src.learner.Manager.RewardManager import *
from src.learner.Manager.ActionManager import *
from src.simlator.GanttChart import *
from master_db.DataInventory import *
from master_db.DB_query import *

from src.model.machine_db import *

import torch
import pickle
import math

class Simulator:
    machine_list = {} #id로 객체 보관
    lot_list = {} #lot id로 객체 보관
    machine_to_factory = {}

    unload_lot_list = {} # 아직 도착하지 못한 job, 처리가 완료된 job 기록
    load_lot_list = {} # 작업 가능한 lot 모음


    number_of_machine = 0
    number_of_job = 0

    #todo 기본 값 세팅 해주어야함
    Q_time_table = {} # {"j0101" : 5 , "j0203" : 5}
    Processing_time_table = {} # {"j0101" : {"M1" : 5, "M2" : 10}, ...}
    job_info = {} #max oper, job type, operation list

    demand_by_planhorizon = {}
    oper_in_list = {}

    done = False #종료조건 판단
    runtime = 0 #시간
    step_number = 0
    event_list = []
    j = 0

    dataSetId = ''
    plan_finish = False
    sorted_event = {"plan_end":3, "job_arrival":2, "DoNothing":2, "track_in_finish":1, "setup_change":1, "NOTHING":1}

    # bucket
    bucket_size = 24 * 7

    # 데이터 프레임 리스트를 피클링하여 파일에 저장
    @classmethod
    def init_simulator(cls, dataSetId):

        cls.dataSetId = dataSetId[0]
        DataInventory.set_db_data(cls.dataSetId)
        cls.get_job_info(cls.dataSetId)
        cls.get_machine(cls.dataSetId)

        cls.get_oper_info(cls.dataSetId)
        cls.get_lot(cls.dataSetId)
        cls.get_mac_status_info(cls.dataSetId)

        with open(f'data_lot_machine_{cls.dataSetId}.pkl', 'wb') as file:
            df_list = [cls.lot_list, cls.machine_list, cls.event_list]
            pickle.dump(df_list, file)
        e = Event(None, "plan_end", "NONE", cls.runtime, Parameters.plan_horizon, "plan_end", "NONE", "NONE", "NONE", 0)
        cls.event_list.append(e)
        cls.get_demand_by_planhorizon()
        cls.lot_categorize()

    @classmethod
    def _init(cls, dataSetId):
        """
        machine에 먼저 접근 -> machine 객체 생성
        job에 접근 -> 비어있는 job dict만들기 , max_oper_of_job , opers_of_job
        setup에 접근 -> setup table만듬
        processing에 접근 -> processing table만듬
        oper에 접근 oper_of_job만듬
        demand에 접근 -> job 객체 생성
        """



        for dataId in dataSetId:
            cls.dataSetId = dataId
            DataInventory.set_db_data(cls.dataSetId)
            cls.get_job_info(cls.dataSetId)
            cls.get_machine(cls.dataSetId)


            cls.get_oper_info(cls.dataSetId)
            cls.get_lot(cls.dataSetId)
            cls.get_mac_status_info(cls.dataSetId)

            with open(f'data_lot_machine_{cls.dataSetId}.pkl', 'wb') as file:
                df_list = [cls.lot_list, cls.machine_list, cls.event_list]
                pickle.dump(df_list, file)
            e = Event(None, "plan_end", "NONE", cls.runtime, Parameters.plan_horizon, "plan_end", "NONE", "NONE", "NONE", 0)
            cls.event_list.append(e)
            cls.lot_list = {}
            cls.event_list = []
            cls.machine_list = {}
            cls.get_demand_by_planhorizon()
    @classmethod
    def reset(cls, dataSetId):
        # 리셋 부분
        cls.dataSetId = dataSetId
        cls.done = False #종료조건 판단
        cls.plan_finish = False
        cls.machine_list = defaultdict(Resource)
        cls.lot_list = defaultdict(Lot)
        cls.runtime = 0  # 시간
        cls.step_number = 0
        cls.event_list = []

        with open(f'data_lot_machine_{cls.dataSetId}.pkl', 'rb') as file:
            loaded_df_list = pickle.load(file)

        cls.lot_list = loaded_df_list[0]
        cls.machine_list = loaded_df_list[1]
        cls.event_list = loaded_df_list[2]

        e = Event(None, "plan_end", "NONE", cls.runtime, Parameters.plan_horizon, "plan_end", "NONE", "NONE", "NONE", 0)
        cls.event_list.append(e)
        cls.get_demand_by_planhorizon()
        s = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job,
                                         cls.demand_by_planhorizon, cls.oper_in_list)

        cls.lot_categorize()
        if Parameters.gantt_on:
            GanttChart.reset()
        return s

    @classmethod
    def get_demand_by_planhorizon(cls):
        bucket_count = math.ceil(Parameters.plan_horizon / cls.bucket_size)

        for job in cls.job_info:
            cls.oper_in_list[job] = 0
            for bucket in range(bucket_count):
                cls.demand_by_planhorizon.setdefault(job, {})[bucket] = 0

        for lot_id in cls.lot_list:
            lot = cls.lot_list[lot_id]
            if lot.duedate <= Parameters.plan_horizon:
                lot_duedate_bucket = math.floor(lot.duedate / cls.bucket_size + 0.001) - 1
                if lot_duedate_bucket != -1:
                    cls.demand_by_planhorizon[lot.job_type][lot_duedate_bucket] += 1
    @classmethod
    def lot_categorize(cls):
        for lot_id in cls.lot_list:
            if cls.lot_list[lot_id].status == ("NOTYET" or "DONE"):
                cls.unload_lot_list[lot_id] = cls.lot_list[lot_id]
            else:
                cls.load_lot_list[lot_id] = cls.lot_list[lot_id]
    @classmethod
    def update_bucket(cls, candidate):
        is_update = False

        if candidate == None:
            return

        lot = candidate[0]
        for bucket in cls.demand_by_planhorizon[lot.job_type]:
            if cls.demand_by_planhorizon[lot.job_type][bucket] > 0:
                cls.demand_by_planhorizon[lot.job_type][bucket] -= 1
                is_update = True
                break

        if is_update == False:
            cls.oper_in_list[lot.job_type] += 1

    @classmethod
    def step(cls, action):
        done = False
        while True:
            machineId = cls.select_machine()
            if machineId == "NONE":
                #이벤트도 비워져 있고, #job들도 다 done이면 종료
                if len(cls.event_list) == 0 and all(cls.lot_list[job].status == "DONE" for job in cls.lot_list):
                    done = True
                    r = 0
                    break
                else:
                    cls.process_event()
                    if cls.plan_finish == True:
                        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job,cls.demand_by_planhorizon, cls.oper_in_list)
                        r = 0
                        done = True
                        break
            else:

                candidate_list = cls.get_candidate(machineId)
                rule_name, candidate = ActionManager.get_lot(candidate_list, action, cls.runtime)

                r, cls.machine_list = RewardManager.get_reward(Parameters.reward_type, machineId, cls.lot_list,
                                                               cls.machine_list, cls.runtime, candidate, cls.demand_by_planhorizon, cls.oper_in_list)
                cls.update_bucket(candidate)
                #print(r)
                s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job,cls.demand_by_planhorizon, cls.oper_in_list)
                cls.get_event(candidate, machineId, rule_name)
                #print(cls.event_list)
                break
        return s_prime, r , done

    @classmethod
    def step2(cls, action_list, eps):
        # action 마스킹용 스텝
        done = False
        while True:
            machineId = cls.select_machine()
            if machineId == "NONE":
                # 이벤트도 비워져 있고, #job들도 다 done이면 종료
                if len(cls.event_list) == 0 and all(cls.lot_list[job].status == "DONE" for job in cls.lot_list):
                    done = True
                    r = 0
                    break
                else:
                    cls.process_event()
                    if cls.plan_finish == True:
                        s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job,
                                                         cls.demand_by_planhorizon, cls.oper_in_list)
                        r = 0
                        action = torch.argmax(action_list).item()
                        done = True
                        break
            else:

                candidate_list = cls.get_candidate(machineId)
                candidate, action, rule_name = ActionManager.get_lot(candidate_list, action_list, cls.runtime)

                r, cls.machine_list = RewardManager.get_reward(Parameters.reward_type, machineId, cls.lot_list,
                                                               cls.machine_list, cls.runtime, candidate,
                                                               cls.demand_by_planhorizon, cls.oper_in_list)
                cls.update_bucket(candidate)
                s_prime = StateManager.get_state(cls.lot_list, cls.machine_list, cls.runtime, cls.number_of_job,
                                                 cls.demand_by_planhorizon, cls.oper_in_list)
                cls.get_event(candidate, machineId, rule_name)
                break

        return s_prime, r, done, action
    @classmethod
    def run(cls, rule):

        while True:
            machineId = cls.select_machine()
            if machineId != "NONE":
                candidate_list = cls.get_candidate(machineId)
                candidate_list = Dispatcher.dispatching_rule_decision(candidate_list, rule, cls.runtime)
                cls.get_event(candidate_list[0], machineId, rule)
            else:
                if len(cls.event_list) == 0 and all(cls.lot_list[lot].status == "DONE" for lot in cls.lot_list):
                    break
                cls.process_event()
                if cls.plan_finish == True:
                    break
                
        
        Flow_time, machine_util, util, makespan, tardiness, lateness, t_max,q_time_true,q_time_false,q_job_t, q_job_f, q_time, rtf = cls.performance_measure()
        if Parameters.gantt_on_check:
            GanttChart.play_gantt()


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
        print("RTF", rtf)
        return Flow_time, util, makespan
    #event = (job_type, operation, machine_type, start_time, end_time, event_type)

    @classmethod
    def gantt_chart(cls):
        if Parameters.gantt_on:
            GanttChart.play_gantt()
    @classmethod
    def load_lot(cls, lot_id):
        if lot_id in cls.unload_lot_list:
            lot = cls.unload_lot_list[lot_id]
            cls.load_lot_list[lot_id] = lot
            del cls.unload_lot_list[lot_id]

    @classmethod
    def unload_lot(cls, lot_id):
        if lot_id in cls.load_lot_list:
            lot = cls.load_lot_list[lot_id]
            cls.unload_lot_list[lot_id] = lot
            del cls.load_lot_list[lot_id]

    @classmethod
    def process_event(cls):
        cls.event_list.sort(key = lambda x:[ x.end_time, cls.sorted_event[x.event_type]], reverse = False)
        event = cls.event_list.pop(0)
        cls.runtime = event.end_time
        if Parameters.log_history:
            #event.send_db(cls.dataSetId, Parameters.simulation_time)
            a=1
        if event.event_type == "job_arrival":
            event.job.arrival()
            cls.load_lot(event.job.id)
        elif event.event_type == "DoNothing":
            event.machine.wake_up_setting()
        elif event.event_type == "plan_end":
            cls.plan_finish = True
        else:
            if event.event_type != "track_in_finish":
                if event.event_type == "setup_change":
                    event_type = "setup"
                    event.machine.complete_setting(event.start_time, event.end_time, event.event_type)  # 기계도 사용가능하도록 변함
                elif event.event_type == "NOTHING":
                    event_type = "NOTHING"
            else:
                #print(event.job)
                event_type = event.job.job_type
                last = event.job.complete_setting(event.start_time, event.end_time ,event.event_type) # 작업이 대기로 변함, 시작시간, 종료시간, event_type
                if last:
                    cls.unload_lot(event.job.id)
                event.machine.complete_setting(event.start_time, event.end_time ,event.event_type) # 기계도 사용가능하도록 변함
            rule = event.rule_name
            step = event.step_num
            start = datetime.fromtimestamp(event.start_time*3600)
            end = datetime.fromtimestamp(event.end_time*3600)
            q_time_diff = event.q_time_diff
            q_time_check = event.q_time_check
            if Parameters.gantt_on or Parameters.log_hitory:
                GanttChart.save_histories(event_type, event.job.id, event.jop, start, end, event.machine.id, rule, step, q_time_diff, q_time_check)

    @classmethod
    def process_event_meta(cls):
        cls.event_list.sort(key=lambda x: x.end_time, reverse=False)
        event = cls.event_list.pop(0)
        cls.runtime = event.end_time
        if event.event_type != "track_in_finish":
            if event.event_type == "setup_change":
                event_type = "setup"
            elif event.event_type == "NOTHING":
                event_type = "NOTHING"
        else:
            # print(event.job)
            event_type = event.job.job_type
            last = event.job.complete_setting(event.start_time, event.end_time,
                                              event.event_type)  # 작업이 대기로 변함, 시작시간, 종료시간, event_type
            event.machine.complete_setting(event.start_time, event.end_time, event.event_type)

    @classmethod
    def assign_setting(cls, job, machine,reservation_time): #job = 1 machine = 1
        q_time_diff = job.assign_setting(machine, cls.runtime)
        machine.assign_setting(job, reservation_time)
        return q_time_diff
    @classmethod
    def select_machine(cls):
        selected_machine = "NONE"
        if cls.runtime >= Parameters.plan_horizon:
            return selected_machine
        for machineId in cls.machine_list:
            if cls.machine_list[machineId].status == "WAIT":
                #todo lot_list를 분류 시켜놓을 필요가 있을듯
                # 예를 들어 현재 stocker에 있을 때 마다 이동? 번거롭더라도 스토커만 확인하는게 맞으니..
                # 종료된 작업이 있는 공간, stocker, 그리고 도착 예정 공간 세개로 나눠서 job을 배치
                for lotId in cls.lot_list: #job 이름과 operation이름 찾기
                    if cls.lot_list[lotId].status != "WAIT": #해당 jop가 작업중일 경우
                        pass
                    #TODO 해당 작업이 해당 기계에서 처리 가능한지 확인해야함
                    elif cls.can_process_oper_in_machine(cls.lot_list[lotId] ,machineId) == False:
                        pass
                    else:
                        selected_machine = machineId
                        break
                if selected_machine != "NONE":
                    break
        return selected_machine
    @classmethod
    def get_least_time_machine(cls, job):
        lot = cls.lot_list[job]
        jobOperId = lot.current_operation_id
        best_machine = ""
        shortest_time = 10000000
        for machineId in cls.machine_list:
            setup_time = cls.machine_list[machineId].get_setup_time(lot.job_type)
            processing_time = cls.Processing_time_table[jobOperId][machineId]
            if processing_time == 0 :
                continue
            start_time = max(cls.machine_list[machineId].last_work_finish_time, lot.act_end_time)
            total_time = setup_time + processing_time + start_time
            if shortest_time > total_time:
                shortest_time = total_time
                best_machine = machineId

        candidate = ([lot, cls.Processing_time_table[jobOperId][best_machine], cls.machine_list[best_machine].get_setup_time(lot.job_type), jobOperId])
        cls.get_event_meta(candidate, best_machine)
        while cls.event_list:
            cls.process_event_meta()
        return best_machine

    @classmethod
    def get_candidate(cls, machineId):
        #todo machine_id와 machine 객체에 대한 구분이 명확해야 할듯
        candidate_list = []
        for lotId in cls.lot_list:
            if cls.lot_list[lotId].status == "WAIT":
                jobOperId = cls.lot_list[lotId].current_operation_id
                setup_time = cls.machine_list[machineId].get_setup_time(cls.lot_list[lotId].job_type)
                if cls.can_process_oper_in_machine(cls.lot_list[lotId], machineId):
                    candidate_list.append([cls.lot_list[lotId],cls.Processing_time_table[jobOperId][machineId], setup_time,jobOperId])

        return candidate_list

    @classmethod
    def get_event(cls, candidate, machineId, rule_name):
        event : Event
        if candidate == None:
            step_num = cls.step_number
            event_end_time = cls.runtime + Parameters.do_nothing_time
            do_nothing_end_time = event_end_time if event_end_time < cls.runtime + Parameters.do_nothing_time \
                else cls.runtime + Parameters.do_nothing_time
            e = Event(None, "", cls.machine_list[machineId], cls.runtime, do_nothing_end_time,
                      "DoNothing",
                      "NONE", step_num, 0, 0)
            cls.machine_list[machineId].do_nothing_setting()
            cls.event_list.append(e)
            cls.step_number += 1
            return

        finish_time = 0
        step_num = cls.step_number
        job, process_time, setup_time, jop = candidate
        if setup_time != 0: #setup event 발생
            finish_time = cls.runtime + setup_time if cls.runtime + setup_time < Parameters.plan_horizon else Parameters.plan_horizon
            e = Event(job, "setup", cls.machine_list[machineId], cls.runtime, finish_time,
                      "setup_change",
                      "NONE", step_num, setup_time, 0)
            cls.event_list.append(e)

        q_time_diff = cls.assign_setting(job, cls.machine_list[machineId],
                                          cls.runtime + setup_time + process_time)
        if finish_time == Parameters.plan_horizon:
            return

        finish_time = cls.runtime + setup_time + process_time if cls.runtime + setup_time + process_time <= Parameters.plan_horizon else Parameters.plan_horizon
        e = Event(job, jop, cls.machine_list[machineId], cls.runtime + setup_time, finish_time,
                  "track_in_finish", rule_name, step_num, setup_time, q_time_diff)
        cls.event_list.append(e)
        cls.step_number +=1

    @classmethod
    def get_event_meta(cls, candidate, machineId):
        step_num = cls.step_number
        job, process_time, setup_time, jop = candidate
        start_time = max(cls.machine_list[machineId].last_work_finish_time, job.act_end_time)
        if setup_time != 0:  # setup event 발생
            e = Event(job, "setup", cls.machine_list[machineId], start_time, start_time + setup_time,
                      "setup_change",
                      "NONE", step_num, setup_time, 0)
            cls.event_list.append(e)
        q_time_diff = cls.assign_setting(job, cls.machine_list[machineId],
                                         start_time + setup_time + process_time)
        e = Event(job, jop, cls.machine_list[machineId], start_time, start_time + setup_time + process_time,
                  "track_in_finish", "meta", step_num, setup_time, q_time_diff)
        cls.event_list.append(e)
        cls.step_number += 1

    @classmethod
    def get_fittness_with_meta_heuristic(cls, job_seq , mac_seq,a=None):
        # chromosome = [[machine seq], [job seq]]
        """
            받은 해를 이용해 이벤트를 생성하고 process event로 처리해야함
            [1,2,1,2,1] ,[2,3,1,4,5]
        """
        for i in range(len(job_seq)):
            lotId = job_seq[i]
            machineId = mac_seq[i]
            jobOperId = cls.lot_list[lotId].current_operation_id
            setup_time = cls.machine_list[machineId].get_setup_time(cls.lot_list[lotId].job_type)
            #print("oper: "+ jobOperId+ " "+"machine:" + machineId +" "+ str(cls.Processing_time_table[jobOperId][machineId]))
            candidate = ([cls.lot_list[lotId], cls.Processing_time_table[jobOperId][machineId], setup_time, jobOperId])
            cls.get_event_meta(candidate, machineId)
            while cls.event_list:
                cls.process_event_meta()

        makespan = 0
        for machine in cls.machine_list:
            if makespan < cls.machine_list[machine].last_work_finish_time:
                makespan = cls.machine_list[machine].last_work_finish_time

        cls.reset(cls.dataSetId)
        return makespan
    @classmethod
    def get_machine(cls, dataSetId):
        #todo 해당 데이터 셋에 해당하는 기계정보를 전부 가져옴 -> 기계 id를
        #todo 기계 정보를 이용해 machine 객체들을 생성함
        # 생성한 객체들을 machine_list에 딕셔너리 형태로 저장함
        machines = DataInventory.get_machine_db_data()
        cls.number_of_machine = len(machines)
        for machine in machines:
            setup_time_table = cls.get_setup_time_table(dataSetId, machine)
            r = Resource(machine.machineId, machine.machineType, setup_time_table, machine.factoryId)
            cls.machine_list[machine.machineId] = r
            cls.machine_to_factory[machine.machineId] = machine.factoryId
    @classmethod
    def get_lot(cls, dataSetId):
        # todo 만약 메타휴리스틱으로 실행시킬 경우에는 lotID를 메타휴리스틱에 적합하도록 설정하는 처리 필요
        jobs = DataInventory.get_demand_db_data()
        for job in jobs:
            if Parameters.meta_ver:
                lot_id = job.jobId
            else:
                lot_id = job.demandId + "-" + job.jobId
            status = ("NOTYET" if job.arrivalData != 0 else "WAIT")
            oper_list = cls.job_info[job.jobId]["oper_list"]
            q_time_table = cls.get_q_time_table_of_opers(oper_list)

            j = Lot(lot_id, job.jobId, cls.job_info[job.jobId]["job_type"] , cls.job_info[job.jobId]["max_oper"]
                    , job.duedate, job.arrivalData, status, oper_list, q_time_table)
            cls.lot_list[lot_id] = j
            if status == "NOTYET":
                e = Event(j, "job_arrival", "NONE", cls.runtime, job.arrivalData, "job_arrival", "NONE", "NONE", "NONE", 0)
                cls.event_list.append(e)
    @classmethod
    def get_mac_status_info(cls, dataSetId):
        mac_status = DataInventory.get_mac_status_db_data()

        if len(mac_status) == 0:
            return

        for mac_row in mac_status:
            machine = cls.machine_list[mac_row.machineId] #machine 객체
            job_id = mac_row.jobId
            j = Lot(mac_row.lotId, mac_row.jobId, cls.job_info[job_id]['job_type'],cls.job_info[job_id]["max_oper"]
                    ,mac_row.dueDate, 0, "WAIT",cls.job_info[job_id]["oper_list"],cls.get_q_time_table_of_opers(cls.job_info[mac_row.jobId]["oper_list"]))
            cls.lot_list[mac_row.lotId] = j

            job = cls.lot_list[mac_row.lotId]
            _ = cls.assign_setting(job, machine, mac_row.finishTime)
            step_num = cls.step_number
            cls.step_number +=1
            e = Event(job, job.current_operation_id, machine, 0, mac_row.finishTime,
                      "track_in_finish", None, step_num, 0, 0)
            cls.event_list.append(e)


    @classmethod
    def get_job_info(cls, dataSetId):
        jobs = DataInventory.get_job_db_data()
        cls.number_of_job = len(jobs)
        #print(type(jobs[0]))
        for job in jobs:
            job_info = {}
            job_info["max_oper"] = job.maxOper
            job_info["job_type"] = job.jobType
            #oper_list = DB_query.get_all_operation_of_job(dataSetId,Oper_db,job.jobId)
            oper_list = DataInventory.sim_data.get_oper_list_by_job(job.jobId)
            job_info["oper_list"] = oper_list
            cls.job_info[job.jobId] = job_info

    @classmethod
    def get_oper_info(cls, dataSetId):
        opers = DataInventory.get_oper_db_data()
        for oper in opers:
            cls.Q_time_table[oper.operId] = oper.operQtime
            for machineId in cls.machine_list:
                """processing_time = DB_query.get_processing_time(dataSetId, ProcessingTime_db,oper.operId,
                                                               cls.machine_list[machineId].machine_type)"""
                processing_time = DataInventory.sim_data.get_processing_time_by_oper_and_machine(oper.operId,
                                                                                                 cls.machine_list[machineId].machine_type)
                if oper.operId not in cls.Processing_time_table:
                    cls.Processing_time_table[oper.operId] = {}
                cls.Processing_time_table[oper.operId][machineId] = processing_time

    @classmethod
    def get_setup_time_table(cls, dataSetId, machine):
        from_to_setup_time_dict = DataInventory.sim_data.get_setup_time_list_by_machine(machine.machineId)
        return from_to_setup_time_dict

    @classmethod
    def get_q_time_table_of_opers(cls, oper_list): # 해당 job의
        q_time_table = {}
        for oper in oper_list:
            q_time_table[oper] = cls.Q_time_table[oper]
        return q_time_table

    @classmethod
    def can_process_oper_in_machine(cls, job, machineId):
        if cls.Processing_time_table[job.current_operation_id][machineId] == 0:
            return False
        else:
            return True
    @classmethod
    def performance_measure(cls):
        q_time_true = 0
        q_time_false = 0
        makespan = cls.runtime
        Flow_time = 0
        Tardiness_time = 0  # new
        Lateness_time = 0  # new
        T_max = 0  # new
        L_max = 0  # new
        value_time_table = []
        full_time_table = []
        machine_util = 0
        util = 0
        q_job_f = 0
        q_job_t = 0
        z = []
        total_q_time_over = 0
        for machineId in cls.machine_list:
            value_added_time, full_time = cls.machine_list[machineId].cal_util()
            value_time_table.append(value_added_time)
            full_time_table.append(full_time)
        util = sum(value_time_table) / sum(full_time_table)

        rtf_target_lot = []
        safe_lot = []
        else_lot = []
        for lotId in cls.lot_list:
            #todo jobFlow time 네이밍
            Flow_time += cls.lot_list[lotId].job_flowtime
            #todo WAIT 같은거 Enum처리
            if cls.lot_list[lotId].status == "WAIT" and cls.lot_list[lotId].duedate <= cls.runtime:
                cls.lot_list[lotId].tardiness_time = cls.runtime - cls.lot_list[lotId].duedate
            if cls.lot_list[lotId].tardiness_time > T_max:
                T_max = cls.lot_list[lotId].tardiness_time
            Tardiness_time += cls.lot_list[lotId].tardiness_time
            Lateness_time += cls.lot_list[lotId].lateness_time
            k = []
            for q in cls.lot_list[lotId].q_time_check_list.values():
                k.append(q)
                if q > 0:
                    q_time_false += 1
                else:
                    q_time_true += 1
            z.append(k)
            if cls.lot_list[lotId].condition == True:
                q_job_t += 1
            else:
                q_job_f += 1
            total_q_time_over += cls.lot_list[lotId].cal_q_time_total()

            if cls.lot_list[lotId].duedate <= makespan:
                rtf_target_lot.append(cls.lot_list[lotId])
                if cls.lot_list[lotId].tardiness_time == 0 and cls.lot_list[lotId].status == "DONE":
                    safe_lot.append(cls.lot_list[lotId])
                else:
                    else_lot.append(cls.lot_list[lotId])
        #rtf 계산 로직
        rtf = round(len(safe_lot) / len(rtf_target_lot) * 100, 2)
        # print(len(safe_lot), len(rtf_target_lot))
        # fig = px.timeline(self.plotlydf, x_start="Start", x_end="Finish", y="Resource", color="Task", width=1000, height=400)
        # fig.show()
        return Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, total_q_time_over, rtf


    @classmethod
    def get_job_seq(cls):
        job_seq = []
        for i in cls.job_info:
            for j in range(cls.job_info[i]["max_oper"]):
                job_seq.append(i)
        return job_seq

    @classmethod
    def get_random_machine(cls, job):
        operId = cls.lot_list[job].current_operation_id
        mac_list = cls.Processing_time_table[operId]
        change_mac_list = []
        for mac, p_time in mac_list.items():
            if p_time != 0:
                change_mac_list.append(mac)

        return random.choice(change_mac_list)




