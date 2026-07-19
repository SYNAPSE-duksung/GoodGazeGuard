# blink 데이터 전처리 코드

## Pipeline Overview

이 파이프라인은 OpenNeuro `ds003838` 데이터셋을 S3에서 스트리밍으로 직접 다운로드하여 전처리부터 라벨링까지 원스톱으로 수행합니다.

```text
[OpenNeuro S3 서버 (ds003838)]
         │  ds003838_pipeline.py (스트리밍 다운로드 & 네트워크 행(Hang) 방지)
         ▼
[Raw Pupil / Behavior 데이터 (tsv)]
         │  blink_features.py (Cho, 2021 방법론 적용)
         ▼
[Blink Features (Blink Entropy, 깜빡임 간격, Lomb-Scargle 스펙트로그램)]
         │  
         ├─► run_pipeline.py (특징 요약본 summary.csv 병합 및 저장)
         │
         └─► labeling.py (개인화 K-means 라벨링)
                  ▼
[최종 라벨링 데이터 (blink_overload_labels_personalized.csv)]

[검증 & 시각화]
         └─► plot_blink_spectrogram.py → 참가자별 스펙트로그램 히트맵 시각화
```


## 파일 구성 및 역할

| 파일명 | 역할 |
| :--- | :--- |
| `run_pipeline.py` | 전체 blink 전처리 파이프라인을 실행하는 메인 스크립트입니다. 다중 스레드(ThreadPoolExecutor)를 통해 피험자 데이터를 병렬 처리합니다. |
| `ds003838_pipeline.py` | HTTPS 퍼블릭 엔드포인트를 통해 데이터를 스트리밍으로 다운로드하는 모듈입니다. |
| `blink_features.py` | Cho (2021) 논문의 방법론(Section 3.2)을 구현하여 blink 신호를 처리하는 핵심 분석 모듈입니다. |
| `labeling.py` | K-means(k=2) 알고리즘을 기반으로 피험자별 '개인화된' 인지 과부하 라벨링을 수행합니다. |
| `plot_blink_spectrogram.py` | 추출된 참가자별 blink 스펙트로그램(`_spectrogram.npy`)을 히트맵 형태로 시각화합니다. |
| `config.py` | 전체 파이프라인이 공유하는 환경 변수(S3 버킷 정보, 타임아웃, 대역폭 등) 설정 파일입니다. |



## 주요 분석 방법론
**1. 눈 깜빡임 특징 추출 (Cho, 2021 기반)**
   - blink_features.py는 눈 깜빡임 이벤트(Blink onset) 시각을 추출하여 다음 지표들을 계산합니다.
   - Lomb-Scargle 스펙트로그램: 불규칙한 눈 깜빡임 간격(IBI) 데이터를 분석하기 위해 Lomb-Scargle periodogram을 사용합니다. 61초 슬라이딩 윈도우를 적용하여 시간에 따른 깜빡임 리듬 변화를 2D 스펙트로그램으로 변환합니다.
   - Blink Entropy (BE): 스펙트로그램 진폭의 히스토그램을 확률 분포로 사용하여 엔트로피를 산출합니다. (수식: $\text{BE}(X) = -\sum p(x_{ij}) \log_2 p(x_{ij})$).
   - Blink Rate: 비교 검증을 위한 표준 지표인 분당 깜빡임 횟수를 계산합니다.

**2. 개인화된 인지 과부하 라벨링 (Personalized K-means)**
   - 단일 기준으로 전체 참가자를 평가하지 않고, labeling.py를 통해 피험자 개개인의 정답률 분포에 맞춰 독립적으로 K-means 클러스터링(k=2)을 수행합니다.
   - 피험자 내에서 상대적으로 정답률이 낮은 군집을 '과부하(Label=1)', 높은 군집을 '정상(Label=0)'으로 자동 분류합니다.
   - 이를 통해 참가자마다 각기 다른 인지 능력 및 과부하 발생 임계점의 개인차를 반영합니다.



## 데이터 다운로드 최적화 
대용량 S3 데이터를 requests로 받아올 때 발생하는 네트워크 무한 대기 현상을 막기 위해 ds003838_pipeline.py에 방어 로직을 적용했습니다.  
- IPv4 강제 적용: 조용히 차단된 IPv6 경로로 인해 DNS 조회가 지연되는 현상을 원천 차단했습니다.  
- 데몬 스레드 및 하드 타임아웃: 다운로드 요청을 별도의 데몬 스레드에 할당하여, 설정된 시간(HARD_TIMEOUT_SEC, 약 16분) 내에 응답이 없으면 멈춰있지 않고 강제 실패 처리 후 다음 피험자로 넘어가도록 설계했습니다.  
- 로컬 캐시 및 수동 다운로드 지원: 한 번 다운로드된 데이터는 s3_cache 폴더에 parquet 포맷으로 저장되어 이후 실행 속도를 대폭 높입니다. 
- 다운로드가 계속 실패하는 피험자는 manual_downloads 폴더에 직접 넣어두면 네트워크를 타지 않고 즉시 로드됩니다.  



## 실행 및 사용 방법
**1. 환경 설정 및 필수 라이브러리 설치**
   pip install numpy pandas scipy scikit-learn matplotlib tqdm requests urllib3

**2. 메인 파이프라인 가동 (전체 피험자 데이터 추출 및 라벨링)**
   - 전체 데이터 처리
   python run_pipeline.py

   - 테스트용으로 소수의 피험자(예: 10명)만 제한해서 돌리고 싶을 때
   python run_pipeline.py --limit 10

   - 결과물 1: blink_features_summary.csv (피험자 윈도우 단위 Blink Features 요약)
   - 결과물 2: blink_overload_labels_personalized.csv (K-means 기반 과부하 라벨링 결과)
   - 결과물 3: blink_spectrograms/ (폴더 내 참가자별 .npy 스펙트로그램 데이터)

**3. Blink 스펙트로그램 시각화**
   - (확인용)파이프라인이 저장한 스펙트로그램 배열 데이터를 2D 히트맵 이미지로 변환합니다.
   - 특정 참가자(예: sub-01) 스펙트로그램 출력
   python plot_blink_spectrogram.py sub-01

   - ID를 생략하면 폴더 내 첫 번째 스펙트로그램을 자동 시각화
   python plot_blink_spectrogram.py

   - 결과물: {sub_id}_spectrogram_heatmap.png (파일 저장 및 화면 출력)
  
## 현재 한계
- **Ground Truth 라벨 부재:** 공식적인 인지 부하 정답이 없어, 피험자별 개인화 K-means 군집화에 의존하여 라벨을 임의로 생성하고 있음
- **군집화와 실제 난이도의 불일치:** 현재 K-means를 k=2(정상/과부하)로 단순화하여 나누고 있어, Task Difficulty의 미세한 연속적 변화를 완전히 대변하지 못할 수 있음
- **데이터 결측치 존재:** 물리적 환경이나 측정 장비 한계로 인해 일부 피험자의 Pupil/Blink 데이터가 아예 누락되는 근본적인 한계가 있음

## 향후 계획

- **최적의 군집 탐색:** PCA 시각화 등을 통해 데이터 분포를 확인하고, K-means의 `n_clusters`를 3개 이상으로 확장하여 세분화된 인지 부하 단계 라벨링 시도
- **모달리티 간 타임스탬프 동기화:** 서로 다른 샘플링 주기를 가진 Blink, Gaze, rPPG 데이터의 시점 일치화
- **Feature-level Fusion:** 다른 팀원 분들이 추출한 특징 데이터와 결합하여 분석 차원 확장
- **Cognitive Load Classification:** 추출된 통합 feature를 바탕으로 정교한 인지 부하 상태 분류 모델 도입
- **Lie Detection Pipeline 구축:** 최종 목표인 인지 부하 기반의 거짓말 판별 머신러닝 분류기 완성
