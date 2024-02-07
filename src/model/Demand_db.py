from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # 모듈 변경
from datetime import datetime  # datetime 모듈 임포트
Base = declarative_base()
class Demand_db(Base):
    __tablename__ = 'Demand'

    dataSetId = Column(String(50), primary_key=True)
    demandId = Column(String(50), primary_key=True)
    jobId = Column(String(50))
    arrivalData = Column(Integer)
    duedate = Column(Integer)
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
    isCreated = Column(DateTime, server_default='CURRENT_TIMESTAMP')