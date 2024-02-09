import numpy as np
from src.Object.Lot import *
from src.learner.common.Hyperparameters import *

class StateManager:
    state_time = 0
    @classmethod
    def get_state(cls, j_list, r_list, cur_runtime, number_of_job, bucket, oper_in_list):
        if Hyperparameters.state_type == "state_12":
            s = cls.set_state_12(j_list, r_list, cur_runtime)
        elif Hyperparameters.state_type == "state_36":
            s = cls.set_state_36(j_list, r_list, cur_runtime, number_of_job)
        elif Hyperparameters.state_type == "action_masking_state":
            s = cls.set_action_masking_state(j_list, r_list, cur_runtime, number_of_job)
        elif Hyperparameters.state_type == "cnn_state":
            s = cls.set_state_cnn(bucket, oper_in_list, cur_runtime)
        elif Hyperparameters.state_type == "cnn":
            s = cls.set_state_cnn_state(bucket, oper_in_list, cur_runtime)
        elif Hyperparameters.state_type == 'default_state':
            s = cls.set_state_default(j_list, r_list, cur_runtime)
        return s

    @classmethod
    def set_state_cnn_state(cls, bucket, oper_in_list, cur_time):
        # cls.state_time = curr_time
        oper_in_list['cur'] = 0

        curr_bucket = cur_time // (7 * 24) if cur_time != 0 else 0
        cur_bucket_list = [0 for i in range(5)]
        curr_bucket = min(4, curr_bucket)
        cur_bucket_list[int(curr_bucket)] = 20

        df_bucket = pd.DataFrame(bucket)
        df_bucket['cur'] = cur_bucket_list
        df_bucket = df_bucket.T

        s_oper_in_list = pd.Series(oper_in_list)
        concat = pd.concat([df_bucket, s_oper_in_list], axis=1)
        normal_concat = concat.div(20)
        s = normal_concat.to_numpy()
        tensor_data = s[np.newaxis, :, :]
        #print(tensor_data)
        return tensor_data

    @classmethod
    def set_state_cnn(cls, bucket, oper_in_list, cur_time):
        #cls.state_time = curr_time
        s = []
        for job, time_bucket_dict in bucket.items():
            for demand_qty in time_bucket_dict.values():
                s.append(demand_qty/20)
        for job_id in oper_in_list:
            qty = oper_in_list[job_id]
            s.append(qty / 20)
        runtime = [0 for i in range(5)]
        curr_bucket = cur_time // (7 * 24) if cur_time != 0 else 0
        curr_bucket = min(4, curr_bucket)
        runtime[int(curr_bucket)] = 1

        s += runtime


        #print(s)
        df = pd.Series(s)
        s = df.to_numpy()

        return s

    @classmethod
    def set_state_default(cls, j_list, r_list, curr_time):
        # stocker의 작업 개수 / 전체 job
        # 처리한 작업 개수 / 전체 job
        # target 설비와 동일 setup 유무
        # of setup
        s = []
        number_of_jobs_wait = 0  # clear
        number_of_jobs_load = 0  # clear
        total_tardiness = 0
        total_flow_time = 0
        number_of_job_done = 0  # clear

        total_job_tardiness_done = 0  # clear
        total_job_q_time_over_done = 0  # clear
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                number_of_jobs_wait += 1
                total_tardiness += j_list[job].cal_tardiness(curr_time)
                total_flow_time += j_list[job].cal_flowtime(curr_time)
            elif j_list[job].status == "PROCESSING":
                number_of_jobs_load += 1
            elif j_list[job].status == "DONE":
                number_of_job_done += 1
                total_job_tardiness_done += j_list[job].tardiness_time

        current_time = curr_time
        total_reservation_time_diff = 0
        max_reservation_time = 0
        for machine in r_list:
            total_reservation_time_diff += r_list[machine].reservation_time - current_time
            if max_reservation_time > r_list[machine].reservation_time:
                max_reservation_time = r_list[machine].reservation_time

        s.append(number_of_jobs_wait/len(j_list))
        s.append(number_of_jobs_load/len(j_list))
        if number_of_jobs_wait == 0:
            for _ in range(2):
                s.append(0)
        else:
            s.append(total_tardiness / number_of_jobs_wait)
            s.append(total_flow_time / number_of_jobs_wait)

        if max_reservation_time == 0:
            s.append(0)
        else:
            s.append(current_time / max_reservation_time)
        s.append(total_reservation_time_diff / len(r_list))

        s.append(number_of_job_done/len(j_list))
        if number_of_job_done == 0:
            s.append(0)
        else:
            s.append(total_job_tardiness_done / number_of_job_done)

        df = pd.Series(s)
        s = df.to_numpy()

        return s

    @classmethod
    def set_state_12(cls, j_list, r_list, cuur_time):
        cls.state_time = cuur_time
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
        number_of_jobs_wait = 0  # clear
        number_of_jobs_load = 0  # clear
        total_remain_operation = 0
        total_tardiness = 0
        total_q_time_over = 0
        total_flow_time = 0
        number_of_job_done = 0  # clear

        total_job_tardiness_done = 0  # clear
        total_job_q_time_over_done = 0  # clear
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                number_of_jobs_wait += 1
                total_remain_operation += j_list[job].remain_operation
                total_tardiness += j_list[job].cal_tardiness(cuur_time)
                total_q_time_over += j_list[job].cal_q_time(cuur_time)
                total_flow_time += j_list[job].cal_flowtime(cuur_time)
            elif j_list[job].status == "PROCESSING":
                number_of_jobs_load += 1
            elif j_list[job].status == "DONE":
                number_of_job_done += 1
                total_job_tardiness_done += j_list[job].tardiness_time
                q_total = j_list[job].cal_q_time_total()
                total_job_q_time_over_done += q_total

        current_time = cuur_time
        total_reservation_time_diff = 0
        max_reservation_time = 0
        for machine in r_list:
            total_reservation_time_diff += r_list[machine].reservation_time - current_time
            if max_reservation_time > r_list[machine].reservation_time:
                max_reservation_time = r_list[machine].reservation_time

        s.append(number_of_jobs_wait)
        s.append(number_of_jobs_load)
        if number_of_jobs_wait == 0:
            for _ in range(4):
                s.append(0)
        else:
            s.append(total_remain_operation / number_of_jobs_wait)
            s.append(total_tardiness / number_of_jobs_wait)
            s.append(total_q_time_over / number_of_jobs_wait)
            s.append(total_flow_time / number_of_jobs_wait)

        s.append(current_time)
        if max_reservation_time == 0:
            s.append(0)
        else:
            s.append(current_time / max_reservation_time)
        s.append(total_reservation_time_diff / len(r_list))

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

    @classmethod
    def set_state_36(cls, j_list, r_list, cur_time, number_of_job_type):
        cls.state_time = cur_time
        """
        len(1) :
        stocker job 개수 / 전체 job
        processing job 개수 / 전체 job
        종료된 job 개수 / 전체 job
        setting setup type length / job type length
        q time over job in stocker / stocker jobs
        q time safe job in stocker / stocker jobs
        
        len(# of machines) :
        각 기계의 util
        각 기계의 r_time / max_r_time
        각 기계의 l_time / max_l_time    
        """
        s = []
        number_of_jobs = len(j_list)
        number_of_machines = len(r_list)
        number_of_job_wait = 0
        number_of_job_load = 0
        number_of_job_done = 0
        q_time_over_job = 0
        q_time_safe_job = 0
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                number_of_job_wait += 1
                check_q_time = j_list[job].check_q_time(cur_time)
                if check_q_time == 0:
                    q_time_over_job += 1
                elif check_q_time == 1:
                    q_time_safe_job += 1
                else:
                    pass
            elif j_list[job].status == "PROCESSING":
                number_of_job_load += 1
            elif j_list[job].status == "DONE":
                number_of_job_done += 1
        s.append(number_of_job_wait/number_of_jobs) #1
        s.append(number_of_job_load/number_of_jobs) #2
        s.append(number_of_job_done/number_of_jobs) #3

        s.append((q_time_over_job/number_of_job_wait) if number_of_job_wait > 0 else 0) #4
        s.append((q_time_safe_job / number_of_job_wait) if number_of_job_wait > 0 else 0) #5

        max_last_work_finish_time = 0
        max_reservation_time = 0
        reservation_list =[]
        lwft_list = [] # last work finish time
        setup_status_list = []
        for machine in r_list:
            util = r_list[machine].cal_util2()
            s.append(util) #7~16
            reservation_list.append(r_list[machine].reservation_time)
            lwft_list.append(r_list[machine].last_work_finish_time)
            setup_status_list.append(r_list[machine].setup_status)
            if max_reservation_time > r_list[machine].reservation_time:
                max_reservation_time = r_list[machine].reservation_time
            if max_last_work_finish_time > r_list[machine].last_work_finish_time:
                max_last_work_finish_time = r_list[machine].last_work_finish_time

        s.append(len(set(setup_status_list))/number_of_job_type) #6

        for i in range(len(reservation_list)):
            s.append((reservation_list[i]/ max_reservation_time) if max_reservation_time >0 else 0) #17~26
            s.append((lwft_list[i] /max_last_work_finish_time) if max_last_work_finish_time > 0 else 0) #27~36

        df = pd.Series(s)
        s = df.to_numpy()

        return s

    @classmethod
    def set_action_masking_state(cls, j_list, r_list, cur_time, number_of_job_type):
        cls.state_time = cur_time
