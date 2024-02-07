from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # 모듈 변경
from datetime import datetime  # datetime 모듈 임포트

Base = declarative_base()
class Mac_Status_db(Base):
    __tablename__ = 'Mac_Status'

    dataSetId = Column(String(50),  primary_key=True)
    machineId = Column(String(50))
    machineType = Column(String(50), primary_key=True)
    status = Column(String(50))
    finishTime = Column(Integer)
    lotId = Column(String(50))
    jobType = Column(String(50))
    jobId = Column(String(50))
    dueDate = Column(Integer)
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
    isCreated = Column(DateTime, server_default='CURRENT_TIMESTAMP')