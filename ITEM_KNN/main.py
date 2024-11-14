from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from held_kerp_b_kmeans_planning import *
import pandas as pd
import pickle
import logging
from typing import List
import uvicorn

# 로거 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# FastAPI 애플리케이션 생성
app = FastAPI(title="Recommendation API", description="추천 시스템 API를 통해 사용자에게 맞춤형 추천을 제공합니다.", version="1.0")

# 모델 로드
with open("./save_models/ITEMKNN-BM25/item_knn_bm25_model.pkl", "rb") as f:
    model = pickle.load(f)

# 추천 API 입출력 데이터 모델 정의
class RecommendationRequest(BaseModel):
    userId: int = Field(..., example=1234, description="추천을 받을 사용자의 ID")
    numPlaceRec: int = Field(..., example=5, ge=1, le=10, description="추천받을 유사 장소의 수")
    placeIds: List[int] = Field(..., example=[2304300002, 2304300001, 2304300004], description="유저가 일정에 추가한 장소 ID list")

class RecommendationResponse(BaseModel):
    userId: int = Field(..., example=1234, description="추천을 요청한 사용자의 ID")
    recPlacedIds: List[int] = Field(..., example=[2305010002, 2304300001, 2304300004, 2304300006, 2304300003], 
                                        description="추천된 아이템 ID 리스트")

# 경로 생성 API 입출력 데이터 모델 정의
class RouteCandidateRequest(BaseModel):
    poi_items: List[str] = Field(..., example=["POI01000000009OTM", "POI01000000000XF8","POI01000000000196", "POI010000000003QP", "POI010000000014QR", "POI010000000002ZX"],
                                    description="경로 생성에 필요한 노드(장소) ID 리스트")
    n_clusters: int = Field(..., example=3, description="클러스터링 개수")
    
    class Config:
        strict = True
        
class RouteClusterReponse(BaseModel):
    paths: List[List] = Field(..., example=[
        [["오지평야", 126.437852644, 36.820256245], ["옥구평야", 126.613907321, 35.959963434], ["오지평야",126.437852644,36.820256245]],
        [["만수앞들", 126.673272889, 37.604314536], ["윗길앞들", 126.810854501, 37.461667414], ["만수앞들", 126.673272889, 37.604314536]],
        [["방구바위들", 128.300164996, 35.899503476], ["남산벌", 128.331245425, 35.294239006], ["방구바위들", 128.300164996, 35.899503476]]])

# 추천 API
@app.post("/recommend", response_model=RecommendationResponse, summary="추천 받기", description="사용자 ID를 입력받아 추천 아이템 리스트를 반환합니다.")
def recommend(request: RecommendationRequest):
    user_id = request.userId
    item_id = request.placeIds
    k = request.numPlaceRec
    
    result = set()
    
    # 모델을 사용해 예측 수행 -> user-based inference
    try:
        recommended_places = model.recommend(user_id=user_id, k=k)
        # item_idx = model.item_ids.index(item_id)
        # user_idx = model.user_ids.index(user_id)

        # score = model.score(user_idx, item_idx)
        logger.info(recommended_places)
        if not recommended_places:  # 추천 결과가 없으면 404 에러 발생
            raise HTTPException(status_code=404, detail="No recommendations found for the given userId.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return JSONResponse(content={"userId": user_id, "recPlaceIds": recommended_places})

# 경로 생성 API
@app.post("/route", response_model=RouteClusterReponse, summary="경로 생성하기", description="장소 ID를 입력받아 경로 클러스터링을 생성합니다.")
def find_optimal_route(request: RouteCandidateRequest):
    poi_list = request.poi_items
    num = request.n_clusters
    
    try: 
        paths = planning(poi_ids=poi_list, n_clusters=num)
        
        if not paths:
            raise HTTPException(status_code=404, detail="No routes found for the given poi list.")
        
        return {"paths": paths}
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# 서버를 실행하기 위한 main 함수
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
