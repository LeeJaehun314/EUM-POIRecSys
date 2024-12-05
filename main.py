from utils import *
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from planning_models.balanced_kmeans.held_kerp_b_kmeans_planning_v2 import *

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
with open("./scoring_models/ITEM_KNN/save_models/ITEMKNN-BM25/item_knn_bm25_model.pkl", "rb") as f:
    model = pickle.load(f)

# 추천 API 입출력 데이터 모델 정의
class RecommendationRequest(BaseModel):
    userId: int = Field(..., example=3, description="추천을 받을 사용자의 ID")
    numPlaceRec: int = Field(..., example=5, ge=1, le=10, description="추천받을 유사 장소의 수")
    placeIds: List[int] = Field(..., example=[1, 2, 3, 4], description="유저가 일정에 추가한 장소 ID list")
    tripAddresses: List[str] = Field(..., example=['서울 마포구', '제주특별자치도 서귀포시'])

class RecommendationResponse(BaseModel):
    userId: int = Field(..., example=1234, description="추천을 요청한 사용자의 ID")
    recPlacedIds: List[int] = Field(..., example=[9, 10, 52], 
                                        description="추천된 아이템 ID 리스트")

# 경로 생성 API 입출력 데이터 모델 정의
class RouteCandidateRequest(BaseModel):
    placeIds: List[List] = Field(..., example=[[123, 126.613907321, 35.959963434], [456, 126.437852644, 36.820256245], 
                                                              [678, 128.300164996, 35.899503476], [789, 120.457896355, 39.820256245]],
                                    description="경로 생성에 필요한 노드(장소) [ID, X 좌표, Y 좌표] 리스트")
    numCluster: int = Field(..., example=2, description="클러스터링 개수(총 여행 일수)")
    
    class Config:
        strict = True
        
class RouteClusterReponse(BaseModel):
    paths: List[List] = Field(..., example=[[123, 456, 123], [678, 789, 678]])

# 추천 API
@app.post("/recommend", response_model=RecommendationResponse, summary="추천 받기", description="사용자 ID를 입력받아 추천 아이템 리스트를 반환합니다.")
async def recommend(request: RecommendationRequest):
    user_id = request.userId
    place_ids = request.placeIds
    k = request.numPlaceRec
    target_addrs = request.tripAddresses
    
    # 모델을 사용해 예측 수행 -> user-based inference
    try:
        user_result = await find_most_similar_users(3, 5)
        recommended_places = await recommend_places_from_filtered_users(user_result, target_addrs, k)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return JSONResponse(content={"userId": user_id, "recPlaceIds": recommended_places})

# 경로 생성 API
@app.post("/route", response_model=RouteClusterReponse, summary="경로 생성하기", description="장소 ID를 입력받아 경로 클러스터링을 생성합니다. 이때 전체 여행 일정들은 숙소에서 출발 및 도착을 보장합니다.")
def find_optimal_route(request: RouteCandidateRequest):
    # 첫 번째 placePosInfo 객체는 숙소로 고정
    place_infos = request.placeIds
    num = request.numCluster
    
    try: 
        paths = planning(place_infos=place_infos, n_clusters=num)
        
        if not paths:
            raise HTTPException(status_code=404, detail="No routes found for the given poi list.")
        
        return JSONResponse(content={"paths": paths})
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))

# 서버를 실행하기 위한 main 함수
if __name__ == "__main__":
    print("Server is running...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
