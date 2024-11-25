import asyncio
import pprint
from sqlalchemy.future import select

from config.database import get_db
from config.models import *

# 타겟 유저와 다른 유저들 간의 유사도 계산하기
async def find_most_similar_users(user_id: int, max_diff=5, cnt=5):
    async for session in get_db():  
        result = await session.execute(select(Preference))
        all_users = result.scalars().all()

        user_pref = next((user for user in all_users if user.id == user_id), None)
        if not user_pref:
            raise ValueError(f"User with ID {user_id} not found.")
        
        other_users = [user for user in all_users if user.id != user_id]

        if not other_users:
            return []

        similarities = []
        for user in other_users:
            diff_cheap = abs(user_pref.cheap_or_expensive - user.cheap_or_expensive)
            diff_plan = abs(user_pref.planned_or_improvise - user.planned_or_improvise)
            diff_popular = abs(user_pref.popular_or_local - user.popular_or_local)
            diff_nature = abs(user_pref.nature_or_city - user.nature_or_city)

            similarity_cheap = 1 - (diff_cheap / max_diff)
            similarity_plan = 1 - (diff_plan / max_diff)
            similarity_popular = 1 - (diff_popular / max_diff)
            similarity_nature = 1 - (diff_nature / max_diff)

            total_similarity = (
                0.25 * similarity_cheap +
                0.25 * similarity_plan +
                0.25 * similarity_popular +
                0.25 * similarity_nature
            )
            similarities.append((user.id, round(total_similarity, 3)))

        most_similar_users = [user_id for user_id, _ in sorted(similarities, key=lambda x: -x[1])][:cnt]

        return most_similar_users


# 타겟 유저가 저장한 장소명 불러오기 - 테스트용
async def find_places_from_target_users(user_id: int):
    async for session in get_db():
        result = await session.execute(
            select(Place.name)
            .join(FolderPlace, FolderPlace.place_id == Place.id)
            .join(Folder, Folder.id == FolderPlace.folder_id)
            .where(Folder.user_id == user_id)
        )
        place_result = result.scalars().all()
        return place_result


# 타겟 유저와 유사도가 높은 사용자들이 저장한 장소 리스트 불러오기
async def recommend_places_from_filtered_users(user_list: list):
    async for session in get_db():
        result = await session.execute(
            select(Place.name)
            .join(FolderPlace, FolderPlace.place_id == Place.id)
            .join(Folder, Folder.id == FolderPlace.folder_id)
            .where(Folder.user_id.in_(user_list))
        )
        name_list = result.scalars().all()
        return name_list
