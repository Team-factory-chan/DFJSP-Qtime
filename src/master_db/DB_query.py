from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # 모듈 변경
from src.common.Parameters import *

class DB_query:
    host = Parameters.db_setting["host"]
    port = Parameters.db_setting["port"]
    user = Parameters.db_setting["user"]
    passwd = Parameters.db_setting["passwd"]
    db = Parameters.db_setting["db"]
    charset = Parameters.db_setting["charset"]

    # SQLAlchemy 연결 엔진 생성
    db_url = f"mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}?charset={charset}"
    engine = create_engine(db_url)

    Session = sessionmaker(bind=engine)
    session = Session()
    Base = declarative_base()

    @classmethod
    def get_all_by_table(cls, dataSetId, table):
        #해당 데이터셋, 해당 테이블의 값을 모두 가져옴
        alls = cls.session.query(table).filter_by(dataSetId=dataSetId).all()
        return alls
    @classmethod
    def get_processing_time(cls, dataSetId, table, operId, machineType):
        # operId, machineType, dataSetId가 모두 일치하는 데이터 가져오기
        row = cls.session.query(table).filter_by(dataSetId=dataSetId, operId=operId,
                                                              machineType=machineType).first()
        return row.processingTime

    @classmethod
    def get_job_type(cls, dataSetId, table ,jobId):
        #jobId의 jobType을 가져옴
        row = cls.session.query(table).filter_by(dataSetId=dataSetId, jobId = jobId).first()
        return row.jobType

    @classmethod
    def get_all_operation_of_job(cls, dataSetId, table, jobId):
        # 해당 job의 모든 operation을 가져옴
        rows = cls.session.query(table).filter_by(dataSetId=dataSetId, jobId=jobId).all()
        oper_list = []
        for row in rows:
            oper_list.append(row.operId)
        return oper_list

    @classmethod
    def get_from_to_setup_time_dict(cls, dataSetId, table, machine, from_job_id):
        #todo job id가 아니라 job type으로 변경 해야함
        rows = cls.session.query(table).filter_by(dataSetId=dataSetId, fromJobType = from_job_id, machineId = machine.machineId).all()
        from_to_setup_time_dict ={}
        for row in rows:
            from_to_setup_time_dict[row.toJobType] = row.setupTime
        return from_to_setup_time_dict

    @classmethod
    def get_total_setup_type(cls, dataSetId, table):
        distinct_job_types = (
            session.query(table.jobType)
            .filter(j1.dataSetId == dataSetId)
            .distinct()
            .all()
        )
        return distinct_job_types

    @classmethod
    def set_event_history(cls, data_to_insert):
        cls.session.add(data_to_insert)
        cls.session.commit()
