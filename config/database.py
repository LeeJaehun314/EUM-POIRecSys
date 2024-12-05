import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Async SQLAlchemy 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True)

# 비동기 세션 생성
async_session = sessionmaker(
    bind=engine, 
    expire_on_commit=False,
    class_=AsyncSession
)

# 데이터베이스 세션을 가져오는 종속성
async def get_db():
    async with async_session() as session:
        yield session
