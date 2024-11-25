import asyncio
import pprint
from sqlalchemy.future import select

from config.database import get_db
from config.models import *

async def find_most_similar_users(user_id: int, max_popular_local=5, cnt=5):
    # cheap_or_expensive, planned_or_improvise, popular_or_local
    async for session in get_db():  
        # Fetch user's preferences
        user_pref = await session.execute(
            select(Preference).where(Preference.id == user_id)
        )
        user_pref = user_pref.scalars().first()
        if not user_pref:
            raise ValueError(f"User with ID {user_id} not found.")

        # Filter users with identical preferred_companions
        other_users = await session.execute(
            select(Preference).where(
                Preference.id != user_id,
                Preference.preferred_companions == user_pref.preferred_companions
            )
        )
        other_users = other_users.scalars().all()

        if not other_users:
            return []

        # Calculate similarities (비동기 유사도 계산 로직)
        similarities = []
        for user in other_users:
            diff_popular = abs(user_pref.popular_or_local - user.popular_or_local)
            normalized_similarity = 1 - (diff_popular / max_popular_local)
            similarities.append((user.id, normalized_similarity))

        # Find the highest similarity score
        most_similar_users = [user for user, _ in sorted(similarities, key=lambda x:-x[1])][:cnt]

        return most_similar_users

if __name__ == "__main__":
    result = asyncio.run(find_most_similar_users(8, 5))
    pprint.pprint(result)