# EUM-POIRecSys
Welcome! This repository is for developing general POI (Position of Interest) Recommendation System with collaboration with EUM-Project.  

# Roadmap
- [X] Setup Github
- [ ] Review RS papers & Choose candidate models
- [ ] Choose evaluation method
- [ ] Make demo scoring model & test platform
- [ ] Make demo planning model & test platform
- [ ] Test three scoring model
- [ ] Test three planning model
- [ ] Conclude the result
- [ ] Make FastAPI interface
- [ ] Deploy to AWS EC2
- [ ] Model serving optimization 
- [ ] Make a MLOps system

# Papers & References

### Github & Paper

| Type | Description | Link | License |
| ---- | ----------- | ---- | ------- |
| Github | **Best Practices on Recommendation Systems** | https://github.com/recommenders-team/recommenders/tree/main?tab=readme-ov-file | MIT |
| Github | **파이썬을 활용한 추천 시스템 구현** | https://github.com/lsjsj92/recommender_system_with_Python | NA |
| Github | A curated list of awesome Recommender System | https://github.com/jihoo-kim/awesome-RecSys | MIT |
| Github | **Neural Collaborative Filtering** | https://github.com/DAC-KHUPID/seoul-date-course-recommendation?tab=readme-ov-file | NA |
| Github | Implementation of 'Personalized POI Recommendation:Spatio-Temporal Representation Learning with Social Tie' | https://github.com/dsj96/PPR-master | MIT |
| Github | Implementation of 'A Diffusion model for POI recommendation' | https://github.com/Yifang-Qin/Diff-POI | Not Specified |
| Github | **POI Paper Archive** | https://github.com/hubojing/POI-Recommendation | NA |
| Github | **Next-POI Paper Archive** | https://github.com/kevin-xuan/Next-POI-Recommendation | NA |
| POI Paper | Translating Embeddings for Modeling Multi-Relational Data | NA | NA |
| POI Paper | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation | http://arxiv.org/abs/2002.02126 | NA |
| POI Paper | Neural Collaborative Filtering | http://arxiv.org/abs/1708.05031 | NA |
| POI Paper | Large Language Models Are Zero-Shot Rankers for Recommender Systems | http://arxiv.org/abs/2305.08845 | NA |
| POI Paper | Large Language Models Meet Collaborative Filtering: An Efficient All-Round LLM-Based Recommender System | http://arxiv.org/abs/2404.11343 | NA |
| POI Paper | **Recent Developments in Recommender Systems: A Survey** | http://arxiv.org/abs/2306.12680 | NA |
| POI Paper | A Survey of Graph Neural Networks for Social Recommender Systems | https://doi.org/10.1145/3661821 | NA |
| POI Paper | An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention | http://arxiv.org/abs/2312.10325 | NA |
| POI Paper | Neighborhood-Enhanced Supervised Contrastive Learning for Collaborative Filtering | https://doi.org/10.1109/TKDE.2023.3317068 | NA |
| POI Paper | Recommender Systems in the Era of Large Language Models (LLMs) | http://arxiv.org/abs/2307.02046 | NA |
| POI Paper | Factorization Machines | https://doi.org/10.1109/ICDM.2010.127 | NA |

# Candidate Models and Test Methods

## Candidates 

### Scoring Model 
- Amazon Personalize [https://aws.amazon.com/ko/personalize/]
- BPR(Cornac/Bayesian Personalized Ranking) [https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb]
- TBD (Maybe NCF?)

### Planning Model 
- Solve as TSP using Dynamic Programming (with Held-Karp Algorithm) [https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm]
- TBD

## Test Methods

### Scoring Model
- NDCG@k [https://velog.io/@whdgnszz1/%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-NDCG]
- Recall@k
- Precision@k

### Planning Model
- NLL (Negative Log Likelihood)
- TBD

# Test Results

### Scoring Model (To be updated)

| Name | NDCG@k | Recall@k | Precision@k |
| ---- | -------- | --- | --- |
| A | 1.0 | 1.0 | 1.0 |
| B | 1.0 | 1.0 | 1.0 |
| C | 1.0 | 1.0 | 1.0 |

### Planning Model

| Name | NLL | TBD | 
| ---- | -------- | --- |
| A | 1.0 | 1.0 |
| B | 1.0 | 1.0 |
| C | 1.0 | 1.0 |
