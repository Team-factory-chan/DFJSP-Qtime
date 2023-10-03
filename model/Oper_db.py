from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # 모듈 변경
from datetime import datetime  # datetime 모듈 임포트

Base = declarative_base()
class Oper_db(Base):
    __tablename__ = 'Oper'

    dataSetId = Column(String(50), ForeignKey('DataSet.dataSetId'), primary_key=True)
    jobId = Column(String(50), ForeignKey('Job.jobId'), primary_key=True)
    operId = Column(String(50), primary_key=True)
    jobType = Column(String(50), ForeignKey('Job.jobType'))
    lastoper = Column(Boolean)
    operDesc = Column(String(255))
    operQtime = Column(Integer)
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
    isCreated = Column(DateTime, server_default='CURRENT_TIMESTAMP')