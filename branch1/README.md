# gaze_pupil_blink_merged.csv + branch1 코드 안내

pupil+gaze+blink 원본 파일들을 하나로 병합한 CSV와, 이걸로 학습하는 branch1 전체 코드 안내임.

## 파일 정보
- 13,529행 (84명 전체 trial)
- key: subject_id + trial_index (0-indexed, pupil/gaze 공통 체계)
- split 컬럼: 기존에 공유한 participant_split.csv 기준 train/valid/test
- label 컬럼: condition(5/9/13)에 대응하는 Low/Medium/High

Group K-Fold 만들 때 subject_id 기준으로 그룹을 나눠야 함 (한 사람의 trial이
train/test에 걸쳐 섞이면 안 됨 -- subject-independent 원칙).

## ⚠️ 주의: 이 CSV는 정제 전 원본임 (데이터 누수 컬럼 포함)
이 파일은 pupil/gaze/blink를 그대로 합치기만 한 것이라, 아래 컬럼들은 그대로 쓰면
trial 길이(condition)를 간접적으로 드러내서 데이터 누수가 남 (우리도 이거 때문에
정확도가 95~100%까지 잘못 나왔던 적 있음):
- `condition` — label을 그대로 결정하는 값, feature로 쓰면 안 됨
- `digit_1~13_zscore` — NaN 패딩 패턴 자체가 trial 길이를 드러냄 (위치별 컬럼 대신 요약 통계로 변환 필요)
- `num_samples`, `scanpath_length`, `fixation_count` — trial 길이에 비례해서 커지는 값
- `n_blinks_ours`, `blink_rate_per_min_ours` — 마찬가지로 trial 길이에 비례

새로 K-fold 코드를 짤 때는 아래 train_branch1_lightgbm.py의 `build_feature_table(df, clean_signal=True)` 함수를 그대로 불러와서 쓰는 걸 추천함 (누수 제거 로직이 이미 다 들어있음).

## src/ 폴더 구성
- `train_branch1_lightgbm.py` — 메인 학습 스크립트. `--merged` 옵션으로 이 CSV를 바로 사용 가능 (실제로 돌려볼 수 있는 건 이거 하나임)
- `clean_blink_dataset.py` — blink 원본 파일 정제(참고용, 이 CSV에는 이미 재계산된 blink 값이 들어있어서 안 써도 됨)
- `compare_blink_contribution.py`, `compare_blink_ours_vs_baseline.py` — blink 추가 효과 비교 (참고용, 이미 결론까지 낸 분석이라 재실행 필수 아님)
- `visualize_branch1_results.py` — 결과 시각화 (참고용)
- `predict_and_export.py` — 최종 확률 예측 CSV 생성, 메타러너 전달용 (참고용)

## 코드 실행
로컬에서는 다음과 같이 실행함. 실행 환경에 맞게 경로만 수정해서 사용:
\`\`\`
python src/train_branch1_lightgbm.py --merged dataset/gaze_pupil_blink_merged.csv --out output/branch1_merged_test --clean-signal
\`\`\`
`--clean-signal`을 빼면 위에서 말한 누수 컬럼이 그대로 포함된 원래(비정상적으로 정확도 높게 나오는) 버전으로 돌아감 -- 비교용으로만 쓸 것.

## 컬럼 구성
- 메타: subject_id, trial_index, block_index, position_in_block, condition, label, remember, split
- pupil: digit_1~13_zscore (raw, NaN 패딩 있음), correct/accuracy/personal_zscore/overload_percentile/overload_label(기존 개인화 라벨)
- gaze: movement_*, scanpath_length, num_samples, gaze_dispersion, dispersion_x/y, center_distance_*, gaze_velocity_*, acceleration_*, fixation_*, hull_area (33개)
- blink: n_blinks_ours, blink_rate_per_min_ours, mean_ibi_ours, std_ibi_ours, blink_entropy_trial_ours (재계산 버전, 커버리지 91~98%)