import asyncio
import pprint
from sqlalchemy.future import select
from sqlalchemy import or_

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


# 타겟 유저와 유사도가 높은 사용자들이 저장한 장소 ID 리스트 불러오기
async def recommend_places_from_filtered_users(user_list: list, target_addr: list, k: int):
    async for session in get_db():
        address_conditions = [
            Place.address.startswith(addr) for addr in target_addr
        ]
        
        # 장소 ID를 조회하면서 주소 필터 추가
        result = await session.execute(
            select(Place.id)
            .join(FolderPlace, FolderPlace.place_id == Place.id)
            .join(Folder, Folder.id == FolderPlace.folder_id)
            .where(
                Folder.user_id.in_(user_list),
                or_(*address_conditions)  # 주소 조건 추가
            )
        )
        
        place_list = result.scalars().all()
        return list(set(place_list))[:k]

# 테스트
if __name__ == "__main__":
    async def main():
        user_result = await find_most_similar_users(3, 5)
        pprint.pprint(f"유사도가 높은 유저 리스트: {user_result}")
        
        target_user = await find_places_from_target_users(3)
        pprint.pprint(f"타겟 유저가 저장한 장소 리스트: {target_user}")
        
        target_addr = ['서울 마포구', '제주특별자치도 서귀포시']
        place_result = await recommend_places_from_filtered_users(user_result, target_addr, 5)
        pprint.pprint(f"유사도가 높은 유저들이 저장한 장소 리스트: {place_result}")

    asyncio.run(main())