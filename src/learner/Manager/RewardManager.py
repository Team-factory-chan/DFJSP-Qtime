class RewardManager:
    @classmethod
    def get_combination_reward_q_over_time_and_makespan(cls, machine, j_list, r_list, curr_time):
        """
        구성 reward : makespan 줄이는 reward * 0.5 + stocker에 대기 중인 q_time_over_time의 총합 * 0.5
        """
        r = 0
        reservation_time = r_list[machine].reservation_time
        last_work_finish_time = r_list[machine].last_work_finish_time
        total_idle = 0
        total_q_time_over = 0
        for resource in r_list:
            if r_list[resource].reservation_time < last_work_finish_time:
                total_idle += (last_work_finish_time - r_list[resource].reservation_time)
                r_list[resource].reservation_time = last_work_finish_time
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                total_q_time_over += j_list[job].cal_q_time(curr_time)

        r -= (0.8 * (reservation_time - last_work_finish_time + total_idle) + 0.2 * total_q_time_over)
        return  r, r_list

    @classmethod
    def get_makespan_reward(cls, machine, j_list, r_list, curr_time):
        """
        구성 reward : makespan 줄이는 reward * 0.5 + stocker에 대기 중인 q_time_over_time의 총합 * 0.5
        """
        r = 0
        reservation_time = r_list[machine].reservation_time
        last_work_finish_time = r_list[machine].last_work_finish_time
        total_idle = 0
        total_q_time_over = 0
        for resource in r_list:
            if r_list[resource].reservation_time < last_work_finish_time:
                total_idle += (last_work_finish_time - r_list[resource].reservation_time)
                r_list[resource].reservation_time = last_work_finish_time
        for job in j_list:  # job 이름과 operation이름 찾기
            if j_list[job].status == "WAIT":
                total_q_time_over += j_list[job].cal_q_time(curr_time)

        r -= reservation_time - last_work_finish_time + total_idle
        return r, r_list
    @classmethod
    def get_rtf_reward2(cls, machine, j_list,r_list, curr_time):
        r = 0
        for job_id in j_list:
            if j_list[job_id].duedate < curr_time and j_list[job_id].status == "WAIT":
                r -= 1
            if j_list[job_id].tardiness_time > 0 and j_list[job_id].status == "DONE":
                r -= 1
            #elif j_list[job_id].status == "DONE" and j_list[job_id].tardiness_time == 0:
                #r += 1
        return  r

    @classmethod
    def get_rtf_reward(cls, candidate,bucket, oper_in_list, curr_time):
        r = 0

        if candidate == None:
            return r

        lot = candidate[0]
        is_demand = False

        for bucket_time in bucket[lot.job_type]:
            if bucket[lot.job_type][bucket_time] > 0:
                is_demand = True
                break

        if is_demand and lot.duedate >= curr_time + 24:
            r += 1
        elif is_demand == False:
            r -= 1

        return r

    @classmethod
    def get_rtf_reward3(cls, candidate,bucket, oper_in_list, curr_time):
        r = 0

        if candidate == None:
            return r

        lot = candidate[0]
        is_demand = False

        for bucket_time in bucket[lot.job_type]:
            if bucket[lot.job_type][bucket_time] > 0:
                is_demand = True
                break

        first_demand_by_job = {}
        demand_qty_by_job = {}
        for job , bucket_time_dict in bucket.items():
            first_demand_by_job[job] = 10
            demand_qty_by_job[job] = 0
            for bucket_time in bucket_time_dict:
                if bucket_time_dict[bucket_time] > 0:
                    first_demand_by_job[job] = bucket_time
                    break

        min_value = min(first_demand_by_job.values())  # 딕셔너리의 최소값 찾기
        min_keys = [key for key, value in first_demand_by_job.items() if value == min_value]

        if lot.job_type in min_keys and min_value != 10:
            r+=1


        if is_demand == False:
            r -= 1


        return r

    @classmethod
    def get_util_reward(cls, machine, candidate):
        r= -candidate[2]/100.0
        return r
    @classmethod
    def get_reward(cls, reward_type,machine, j_list, r_list, curr_time, candidate, bucket, oper_in_list):
        if candidate == None:
            r = 0
            # Donothing action에 대한 패널티 부여
        if reward_type == COMBINATION_Q_OVER_TIME_AND_MAKESPAN:
            r, r_list = cls.get_combination_reward_q_over_time_and_makespan(machine, j_list, r_list, curr_time)
        elif reward_type == MAKESPAN:
            r, r_list = cls.get_makespan_reward(machine,j_list,r_list,curr_time)
        elif reward_type == "rtf2":
            r = cls.get_rtf_reward2(machine,j_list,r_list,curr_time)
        elif reward_type == RTF:
            r = cls.get_rtf_reward(candidate, bucket, oper_in_list, curr_time)
        elif reward_type == "util":
            r = cls.get_util_reward(machine,candidate)

        return r, r_list

COMBINATION_Q_OVER_TIME_AND_MAKESPAN = "combination_reward_q_over_time_and_makespan"
MAKESPAN = "makespan"
RTF = "rtf"