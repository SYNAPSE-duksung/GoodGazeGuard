import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "branch1" / "src"))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

import lightgbm as lgb
from train_branch1_lightgbm import build_feature_table

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report

# 기존 코드 재사용
from train_branch1_lightgbm import (
    load_and_merge,
    build_feature_table,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--merged", required=True)      # 읽어올 CSV
    parser.add_argument("--out", default="output/group_kfold")      # 결과 저장 폴더
    parser.add_argument("--clean-signal", action="store_true")      # 데이터 누수 제거한 feature만 사용

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ##################################################
    # Load
    ##################################################

    print("Loading merged dataset...")
    df = pd.read_csv(args.merged)

    if df["remember"].dtype == object:
        df["remember"] = df["remember"].map(
            {"True": True, "False": False}
        )

    ##################################################
    # Feature
    ##################################################

    X, y, feature_cols = build_feature_table(
        df,
        clean_signal=args.clean_signal
    )

    # group 생성 시 subject 기준으로 나눔
    groups = df["subject_id"]

    ##################################################
    # Group KFold
    ##################################################

    gkf = GroupKFold(n_splits=5)

    fold_scores = []

    feature_importances = []

    # 전체 Confusion Matrix 위해 변수 추가
    all_true = []
    all_pred = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups)
    ):

        print("=" * 60)
        print(f"Fold {fold+1}")
        print("=" * 60)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        train_data = lgb.Dataset(
            X_train,
            label=y_train
        )

        test_data = lgb.Dataset(
            X_test,
            label=y_test,
            reference=train_data
        )

        # 3-class classification
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",

            "learning_rate": 0.03,
            "num_leaves": 31,
            "feature_fraction": 0.8,    # 매 트리를 만들 때 전체 feature의 80%만 랜덤하게 사용
            "bagging_fraction": 0.8,    # 데이터 샘플을 랜덤하게 80%만 사용
            "bagging_freq": 5,          # 5번마다 랜덤 샘플링 다시
            "min_data_in_leaf": 30,     # 잎 하나에 최소 30개의 데이터

            "seed": 42,
        }

        # 모델 학습
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(50),
            ],
        )

        pred_prob = model.predict(
            X_test,
            num_iteration=model.best_iteration
        )
        # 가장 큰 확률을 label로 만듦
        pred = np.argmax(pred_prob, axis=1)

        acc = accuracy_score(y_test, pred)

        fold_scores.append(acc)

        all_true.extend(y_test)
        all_pred.extend(pred)

        print(f"\nFold {fold+1} Accuracy : {acc:.4f}")

        print(
            classification_report(
                y_test,
                pred,
                target_names=[
                    "Low",
                    "Medium",
                    "High"
                ]
            )
        )

        feature_importances.append(
            model.feature_importance(
                importance_type="gain"
            )
        )

    ##################################################
    # Result
    ##################################################

    print("\n")
    print("=" * 60)
    print("Final Result")
    print("=" * 60)

    print(f"Fold Scores : {fold_scores}")
    print(f"Mean Accuracy : {np.mean(fold_scores):.4f}")
    print(f"Std Accuracy : {np.std(fold_scores):.4f}")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": np.mean(feature_importances, axis=0),
        "importance_std": np.std(feature_importances, axis=0)
    }).sort_values("importance_mean", ascending=False)

    importance.to_csv(
        out_dir / "feature_importance.csv",
        index=False
    )

    pd.DataFrame({
        "fold": np.arange(1, 6),
        "accuracy": fold_scores
    }).to_csv(
        out_dir / "fold_accuracy.csv",
        index=False
    )

    print("\nSaved Results.")

    # -----------------------------
    # Feature Importance Plot
    # -----------------------------

    top15 = importance.head(15)

    plt.figure(figsize=(8,6))
    plt.barh(
        top15["feature"][::-1],
        top15["importance_mean"][::-1]
    )
    plt.xlabel("Importance")
    plt.title("Top 15 Feature Importance (Mean over 5 folds)")
    plt.tight_layout()
    plt.savefig(
        out_dir / "feature_importance.png",
        dpi=300
    )
    plt.close()

    # -----------------------------
    # Fold Accuracy
    # -----------------------------

    plt.figure(figsize=(6,4))
    plt.bar(
        range(1,6),
        fold_scores
    )
    plt.ylim(0.6,0.85)
    plt.xticks(range(1,6))
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Group K-Fold Accuracy")
    for i, score in enumerate(fold_scores):
        plt.text(
            i+1,
            score+0.005,
            f"{score:.3f}",
            ha="center"
        )
    plt.tight_layout()
    plt.savefig(
        out_dir / "fold_accuracy.png",
        dpi=300
    )
    plt.close()

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    cm = confusion_matrix(
        all_true,
        all_pred
    )
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low","Medium","High"],
        yticklabels=["Low","Medium","High"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (All 5 Folds)")
    plt.tight_layout()
    plt.savefig(
        out_dir / "confusion_matrix.png",
        dpi=300
    )
    plt.close()


if __name__ == "__main__":
    main()