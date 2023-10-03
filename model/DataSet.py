from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  # 모듈 변경
from datetime import datetime  # datetime 모듈 임포트

Base = declarative_base()
class DataSet(Base):
    __tablename__ = 'DataSet'

    dataSetId = Column(String(50), primary_key=True)
    dataDesc = Column(String(255))
    createUser = Column(String(50))
    isUpdated = Column(DateTime, default=datetime.now)  # 기본값 설정
