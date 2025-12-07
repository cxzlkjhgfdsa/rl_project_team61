# RL Project

## 1. 프로잭트 개요

Proximal Policy Optimization Algorithm, Schulman et al, 2017 논문 재현실험

-   알고리즘 : Proximal Policy Optimizatio (PPO)
-   환경 : Gymnasium Mujoco HalfCheetah-v5
-   목표 : 논문 재현 및 커버되지 않은 파라미터 실험

## 2. 개발 환경 및 실행방법

**Python 및 라이브러리 버전**

-   Python : 3.10.16
-   Pytorch : 2.7.1
-   Gymnasium : 1.2.2
-   Numpy : 2.1.3

**설치 , 실행 방법**

```jsx
# 라이브러리 설치
pip install torch gymnasium numpy matplotlib

# 실행
python {code file name }
```

code file 은 다른 surrogate object 에 따라 총 4가지 file 로 구분되어있음

-   non clipping : `non_clipping.py`
-   clipping : `clipping.py`
-   fixed kl : `fixed_kl.py`
-   adaptive kl : `adaptive_kl.py`

**HyperParameters**

코드 내 `HYPERPARAMS` Dictionary 를 통해 HyperParameter 변경 가능, 본 프로젝트에서는 아래 3가지 값만 변경하며 실험을 진행함

-   epsilon
-   beta
-   kl_target

## 3. 결과 확인 및 로깅

**로깅**

-   매 update 마다 [ 학습 step, 최근 10개 episode 평균 reward ] 가 console 에 출력됨
-   adaptive kl 실험해서는 추가적으로 [ 현재 beta 값, 평균 KL Divergence 값 ] 가 console 에 출력됨
-   각 시드별 학습 종료시 최근 100개 episode 의 평균 reward가 console 에 출력됨

**결과 그래프**

-   학습이 완료되면, 3개 시드의 평균 보상과 신뢰구간 ( +- 표준편차 ) 를 나타내는 그래프파일이 자동으로 생성됨
