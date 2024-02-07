class simulator_manager:
    machine_list = {} #id로 객체 보관
    lot_list = {} #lot id로 객체 보관
    factory_list = {}

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

    j = 0
    dataSetId = ''

    plan_finish = False
    sorted_event = {"plan_end":3, "job_arrival":2, "DoNothing":2, "track_in_finish":1, "setup_change":1, "NOTHING":1}

    # bucket
    bucket_size = 24 * 7

    @classmethod
    def init(cls, dataSetId):
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
    def get_machine(cls, dataSetId):
        # todo 해당 데이터 셋에 해당하는 기계정보를 전부 가져옴 -> 기계 id를
        # todo 기계 정보를 이용해 machine 객체들을 생성함
        # 생성한 객체들을 machine_list에 딕셔너리 형태로 저장함
        machines = DataInventory.get_machine_db_data()
        cls.number_of_machine = len(machines)
        for machine in machines:
            setup_time_table = cls.get_setup_time_table(dataSetId, machine)
            r = Resource(machine.machineId, machine.machineType, setup_time_table)
            cls.machine_list[machine.machineId] = r

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

            j = Lot(lot_id, job.jobId, cls.job_info[job.jobId]["job_type"], cls.job_info[job.jobId]["max_oper"]
                    , job.duedate, job.arrivalData, status, oper_list, q_time_table)
            cls.lot_list[lot_id] = j
            if status == "NOTYET":
                e = Event(j, "job_arrival", "NONE", cls.runtime, job.arrivalData, "job_arrival", "NONE", "NONE", "NONE",
                          0)
                cls.event_list.append(e)

    @classmethod
    def get_mac_status_info(cls, dataSetId):
        mac_status = DataInventory.get_mac_status_db_data()

        if len(mac_status) == 0:
            return

        for mac_row in mac_status:
            machine = cls.machine_list[mac_row.machineId]  # machine 객체
            job_id = mac_row.jobId
            j = Lot(mac_row.lotId, mac_row.jobId, cls.job_info[job_id]['job_type'], cls.job_info[job_id]["max_oper"]
                    , mac_row.dueDate, 0, "WAIT", cls.job_info[job_id]["oper_list"],
                    cls.get_q_time_table_of_opers(cls.job_info[mac_row.jobId]["oper_list"]))
            cls.lot_list[mac_row.lotId] = j

            job = cls.lot_list[mac_row.lotId]
            _ = cls.assign_setting(job, machine, mac_row.finishTime)
            step_num = cls.step_number
            cls.step_number += 1
            e = Event(job, job.current_operation_id, machine, 0, mac_row.finishTime,
                      "track_in_finish", None, step_num, 0, 0)
            cls.event_list.append(e)

    @classmethod
    def get_job_info(cls, dataSetId):
        jobs = DataInventory.get_job_db_data()
        cls.number_of_job = len(jobs)
        # print(type(jobs[0]))
        for job in jobs:
            job_info = {}
            job_info["max_oper"] = job.maxOper
            job_info["job_type"] = job.jobType
            # oper_list = DB_query.get_all_operation_of_job(dataSetId,Oper_db,job.jobId)
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
                                                                                                 cls.machine_list[
                                                                                                     machineId].machine_type)
                if oper.operId not in cls.Processing_time_table:
                    cls.Processing_time_table[oper.operId] = {}
                cls.Processing_time_table[oper.operId][machineId] = processing_time

    @classmethod
    def get_setup_time_table(cls, dataSetId, machine):
        from_to_setup_time_dict = DataInventory.sim_data.get_setup_time_list_by_machine(machine.machineId)
        return from_to_setup_time_dict

    @classmethod
    def get_q_time_table_of_opers(cls, oper_list):  # 해당 job의
        q_time_table = {}
        for oper in oper_list:
            q_time_table[oper] = cls.Q_time_table[oper]
        return q_time_table