from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import logging
from typing import List
import uvicorn

logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI(title="Recommendation API", description="추천 시스템 API를 통해 사용자에게 맞춤형 추천을 제공합니다.", version="1.0")

# 모델 로드
with open("./save_models/ITEMKNN-BM25/item_knn_bm25_model.pkl", "rb") as f:
    model = pickle.load(f)

# 입력 데이터 모델 정의
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="e000004", description="추천을 받을 사용자의 ID")
    num: int = Field(..., example=5, ge=1, le=10, description="추천받을 아이템의 수")
    item_id: int = Field(..., example=2304300002)

class RecommendationResponse(BaseModel):
    user_id: str = Field(..., example="e000004", description="추천을 요청한 사용자의 ID")
    recommended: List[int] = Field(..., example=[2305010002, 2304300001, 2304300004, 2304300006, 2304300003], 
                                        description="추천된 아이템 ID 리스트")
    score: float = Field(..., example=3.49340373257926)

@app.post("/recommend", response_model=RecommendationResponse, summary="추천 받기", description="사용자 ID를 입력받아 추천 아이템 리스트를 반환합니다.")
def recommend(request: RecommendationRequest):
    user_id = request.user_id
    item_id = request.item_id
    k = request.num
    
    # 모델을 사용해 예측 수행
    try:
        recommended_places = model.recommend(user_id=user_id, k=k)
        item_idx = model.item_ids.index(item_id)
        user_idx = model.user_ids.index(user_id)

        score = model.score(user_idx, item_idx)
        logger.info(recommended_places)
        if (not recommended_places) or (not score):  # 추천 결과가 없으면 404 에러 발생
            raise HTTPException(status_code=404, detail="No recommendations found for the given user_id.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"user_id": user_id, "score": score, "recommended": recommended_places}

# 서버를 실행하기 위한 main 함수
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
