class SimDataInven:
    def __init__(self):
        self.oper_list_by_job = {}
        self.setup_time_list_by_machine = {}
        self.processing_time_by_oper_and_machine = {}

    def set_oper_list_by_job(self, value):
        self.oper_list_by_job = value
    def set_setup_time_list_by_machine(self, value):
        self.setup_time_list_by_machine = value
    def set_processing_time_by_oper_and_machine(self, value):
        self.processing_time_by_oper_and_machine = value

    def get_oper_list_by_job(self, job):
        return self.oper_list_by_job[job]
    def get_setup_time_list_by_machine(self, machine):
        return self.setup_time_list_by_machine[machine]

    def get_processing_time_by_oper_and_machine(self, operid, machineid):
        return self.processing_time_by_oper_and_machine[(operid,machineid)]
