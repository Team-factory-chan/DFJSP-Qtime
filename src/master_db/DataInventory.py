from master_db.DB_query import *

import os
import pickle

from src.model.Job_db import *
from src.model.Oper_db import *
from src.model.ProcessingTime_db import *
from src.model.Setup_db import *
from src.model.Demand_db import *
from src.model.machine_db import *
from src.model.mac_status_db import *
from src.model.Factory_db import *
from master_db.SimDataInven import *

class DataInventory:
    master_data = {}        # dataFrame 키, value로 data저장
    sim_data = SimDataInven()
    dict_data = {}      # dict데이터
    dataset_id = "MK01"
    db_dict = {"Machine_db": Machine_db, "Demand_db":Demand_db, "Job_db":Job_db,
               "Oper_db":Oper_db,"ProcessingTime_db":ProcessingTime_db,"Setup_db":Setup_db, "MacStatus_db": Mac_Status_db,
               "Factory_db":Factory_db}
    @classmethod
    def set_db_data(cls, data_id):
        cls.dataset_id = data_id
        file_path = f'{pathConfig.absolute_path}/{cls.dataset_id}_db_data.pkl'
        if os.path.exists(file_path):

            with open(file_path, 'rb') as file:
                data_list = pickle.load(file)
                cls.master_data = data_list[0]
                cls.sim_data = data_list[1]
            print(f"피클 파일이 존재합니다.")
        else:
            for key, value in cls.db_dict.items():
                data_df = DB_query.get_all_by_table(cls.dataset_id, value)
                cls.master_data[key] = data_df
            cls.set_sim_data()
            data_list = [cls.master_data, cls.sim_data]

            with open(file_path, 'wb') as file:
                pickle.dump(data_list, file)

            print("db 데이터 전부 저장")
    @classmethod
    def set_sim_data(cls):
        cls._set_oper_list()
        cls._set_setup_time_list()
        cls._set_processing_time_table()
        print("Sim data setting")
    @classmethod
    def _set_oper_list(cls):
        jobs = cls.master_data["Job_db"]
        oper_dict = {}
        for job in jobs:
            oper_list = DB_query.get_all_operation_of_job(cls.dataset_id, Oper_db, job.jobId)
            oper_dict[job.jobId] = oper_list
        cls.sim_data.set_oper_list_by_job(oper_dict)
    @classmethod
    def _set_setup_time_list(cls):
        machines = cls.master_data["Machine_db"]
        setup_dict = {}
        for machine in machines:
            setup_time_table = cls._get_setup_time_table(cls.dataset_id, machine)
            setup_dict[machine.machineId] = setup_time_table
        cls.sim_data.set_setup_time_list_by_machine(setup_dict)
    @classmethod
    def _set_processing_time_table(cls):
        opers = cls.master_data["Oper_db"]
        machines = cls.master_data["Machine_db"]
        processing_time_dict = {}
        for oper in opers:
            for machine in machines:
                processing_time = DB_query.get_processing_time(cls.dataset_id, ProcessingTime_db,oper.operId,
                                                               machine.machineType)
                processing_time_dict[(oper.operId, machine.machineType)] = processing_time

        cls.sim_data.set_processing_time_by_oper_and_machine(processing_time_dict)

    @classmethod
    def _get_setup_time_table(cls, dataSetId, machine):
        machine_setup_time_dict = {}
        jobs = cls.master_data["Job_db"]
        for from_job_id in jobs:
            from_to_setup_time_dict = DB_query.get_from_to_setup_time_dict(dataSetId, Setup_db, machine, from_job_id.jobId)
            machine_setup_time_dict[from_job_id.jobId] = from_to_setup_time_dict
        return machine_setup_time_dict
    @classmethod
    def get_job_db_data(cls):
        return cls.master_data["Job_db"]

    @classmethod
    def get_machine_db_data(cls):
        return cls.master_data["Machine_db"]

    @classmethod
    def get_oper_db_data(cls):
        return cls.master_data["Oper_db"]

    @classmethod
    def get_processing_db_data(cls):
        return cls.master_data["ProcessingTime_db"]

    @classmethod
    def get_setup_db_data(cls):
        return cls.master_data["Setup_db"]

    @classmethod
    def get_demand_db_data(cls):
        return cls.master_data["Demand_db"]

    @classmethod
    def get_factory_db_data(cls):
        return cls.master_data['Factory_db']

    @classmethod
    def get_mac_status_db_data(cls):
        return cls.master_data['MacStatus_db'] if "MacStatus_db" in cls.master_data else []