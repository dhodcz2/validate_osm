import datetime
import sqlite3
import sqlalchemy
from sqlalchemy.orm import declarative_base

Base = declarative_base()

from sqlalchemy import Column, Integer, String, Float, Date


class Distance(Base):
    __tablename__ = 'distance'

    bx = Column(Float, key=True)
    by = Column(Float, key=True)
    distance = Column(Float, )
