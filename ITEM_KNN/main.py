from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import logging

logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI()

# 모델 로드
with open("./save_models/ITEMKNN-BM25/item_knn_bm25_model.pkl", "rb") as f:
    model = pickle.load(f)

# 입력 데이터 모델 정의
class RecommendationRequest(BaseModel):
    user_id: str
    item_id: int
    num: int

@app.post("/recommend")
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
