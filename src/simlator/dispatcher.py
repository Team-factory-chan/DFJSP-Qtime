from collections import Counter
class Dispatcher:
    @classmethod
    def dispatching_rule_decision(cls, candidate_list, rule_name, curr_time):
        cls.time = curr_time

        if rule_name == "SPT":
            candidate_list = cls.dispatching_rule_SPT(candidate_list)
        elif rule_name == "SSU":
            candidate_list = cls.dispatching_rule_SSU(candidate_list)
        elif rule_name == "SPTSSU":
            candidate_list = cls.dispatching_rule_SPTSSU(candidate_list)
        elif rule_name == "MOR":
            candidate_list = cls.dispatching_rule_MOR(candidate_list)
        elif rule_name == "LOR":
            candidate_list = cls.dispatching_rule_LOR(candidate_list)
        elif rule_name == "EDD":
            candidate_list = cls.dispatching_rule_EDD(candidate_list)
        elif rule_name == "MST":
            candidate_list = cls.dispatching_rule_MST(candidate_list)
        elif rule_name == "FIFO":
            candidate_list = cls.dispatching_rule_FIFO(candidate_list)
        elif rule_name == "LIFO":
            candidate_list = cls.dispatching_rule_LIFO(candidate_list)
        elif rule_name == "CR":
            candidate_list = cls.dispatching_rule_CR(candidate_list)
        elif rule_name == "None":
            candidate_list = cls.dispatching_rule_NONE(candidate_list)
        elif rule_name == "SQT":
            candidate_list = cls.dispatching_rule_SQT(candidate_list,curr_time)
        elif rule_name == "NewRule1":
            candidate_list = cls.dispatching_rule_NewRule1(candidate_list)
        elif rule_name == "NewRule2":
            candidate_list = cls.dispatching_rule_NewRule2(candidate_list)


        return candidate_list

    def dispatching_rule_SPT(cls, candidate_list):
        #candidate_list = [job, processing_time, setup_time, finish time]
        candidate_list.sort(key=lambda x: x[1], reverse=False)
        #return job, processing_time, setup_time, finish_time
        return candidate_list

    def dispatching_rule_SSU(cls, candidate_list):

        candidate_list.sort(key=lambda x: [ x[2], x[0].duedate], reverse=False)

        return candidate_list

    def dispatching_rule_SPTSSU(cls, candidate_list):
        candidate_list.sort(key=lambda x: [ x[1]+x[2], x[0].duedate] , reverse=False)

        return candidate_list

    def dispatching_rule_MOR(cls, candidate_list):
        candidate_list.sort(key=lambda x: x[0].remain_operation, reverse=True)

        return candidate_list

    def dispatching_rule_LOR(cls, candidate_list):

        candidate_list.sort(key=lambda x: x[0].remain_operation, reverse=False)
        return candidate_list

    def dispatching_rule_EDD(cls, candidate_list):
        candidate_list.sort(key=lambda x: [ x[0].duedate, x[2]] , reverse=False)

        return candidate_list

    def dispatching_rule_NewRule1(cls, candidate_list):
        # candidate_list.sort(key=lambda x: x[0].duedate - self.time - x[1], reverse=False)
        #candidate_list.sort(key = lambda x: [x[2]] , reverse = False)

        # 후보 1 : edd로 하고 중복은 SPT로
        candidate_list.sort(key = lambda x: [x[0].duedate, x[1]] , reverse = False)

        return candidate_list

    def dispatching_rule_NewRule2(cls, candidate_list):
        # candidate_list.sort(key=lambda x: x[0].duedate - self.time - x[1], reverse=False)
        #candidate_list.sort(key = lambda x: [x[2]] , reverse = False)

        # 후보 1 : edd로 하고 중복은 SPT로
        candidate_list.sort(key = lambda x: [x[0].duedate, x[1]] , reverse = False)
        # 후보 2 : 처리 가능한 작업
        #candidate_list.sort(key = lambda x: x[0].duedate , reverse = False)
        #if candidate_list[0]
        #if candidate_list[0][0].duedate - self.time > 20:
        #    candidate_list.sort(key=lambda x: x[1], reverse=False)
        demands = [candidate[0].job_type for candidate in candidate_list]
        demand_counts= Counter(demands)

        min_due_date = min(candidate[0].duedate for candidate in candidate_list)

        def custom_sort(candidate):
            #return ( min_due_date if candidate[0].duedate == min_due_date else candidate[0].duedate , demand_counts[candidate[0].job_type]* candidate[1] if demand_counts[candidate[0].job_type]!=None else 0 )
            #return ( candidate[0].duedate + candidate[1] )
            return (candidate[0].duedate + (candidate[1] * len(demand_counts)) + (cls.time / demand_counts[candidate[0].job_type]) )

        candidate_list = sorted(candidate_list, key = custom_sort, reverse=False)

        return candidate_list

    def dispatching_rule_CR(cls, candidate_list):
        candidate_list.sort(key=lambda x: (x[0].duedate - cls.time) / x[1], reverse=False)
        return candidate_list

    def dispatching_rule_MST(cls, candidate_list):
        candidate_list.sort(key=lambda x: [ x[0].duedate - cls.time - x[1], x[2]], reverse=False)
        return candidate_list

    def dispatching_rule_FIFO(cls, candidate_list):

        candidate_list.sort(key=lambda x: [x[0].job_arrival_time, x[2]], reverse=False)
        return candidate_list

    def dispatching_rule_LIFO(cls, candidate_list):
        candidate_list.sort(key=lambda x: [x[0].job_arrival_time, x[2]], reverse=True)
        return candidate_list

    def dispatching_rule_SQT(cls, candidate_list,curr_time):
        candidate_list.sort(key=lambda x: [-x[0].cal_q_time(curr_time) ,x[0].job_arrival_time], reverse = False)
        return candidate_list

    def dispatching_rule_NONE(cls, candidate_list):
        candidate_list = []
        return candidate_list
