from sqlalchemy import create_engine, Column, String, DateTime, Integer, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # 모듈 변경
from datetime import datetime  # datetime 모듈 임포트

Base = declarative_base()
class Setup_db(Base):
    __tablename__ = 'Setup'

    dataSetId = Column(String(50),  primary_key=True)
    machineId = Column(String(50),  primary_key=True)
    machineType = Column(String(50), primary_key=True)
    fromJobType = Column(String(50), primary_key=True)
    toJobType = Column(String(50),  primary_key=True)
    setupTime = Column(Integer)
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
    isCreated = Column(DateTime, server_default='CURRENT_TIMESTAMP')