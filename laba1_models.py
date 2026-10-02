# -*- coding: utf-8 -*-
"""
Лабораторная работа №1 — часть 2: регрессия и PCA.

Train/test, стандартизация, OLS / Ridge / Lasso (GridSearchCV),
графики прогнозов и остатков, PCA (каменистая осыпь), сравнение метрик.
Графики сохраняются в figures/, веса — в csv/json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

TARGET = "Медианная цена дома"
ALPHA_GRID = {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]}


def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    return rmse, r2, mape


def label_component(series: pd.Series) -> tuple[str, str]:
    s = series.abs().sort_values(ascending=False)
    top = s.head(3)
    names = list(top.index)
    signs = ["+" if series[n] >= 0 else "-" for n in names]
    detail = ", ".join(
        f"{signs[i]}{names[i]} ({series[names[i]]:+.2f})" for i in range(len(names))
    )
    title = " / ".join(names[:2])
    return title, detail


def load_prepared() -> pd.DataFrame:
    prepared = ROOT / "housing_prepared.csv"
    if prepared.exists():
        print(f"Загружен подготовленный датасет: {prepared}")
        return pd.read_csv(prepared)

    # если analysis ещё не запускали — подготовим здесь
    from laba1_analysis import load_and_prepare

    _, df_model = load_and_prepare()
    df_model.to_csv(prepared, index=False, encoding="utf-8-sig")
    return df_model


def split_and_scale(df_model: pd.DataFrame):
    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, признаков: {len(feature_names)}")
    return (
        feature_names,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        kf,
        scaler,
    )


def tune_regularization(X_train_scaled, y_train, kf):
    ridge_search = GridSearchCV(
        Ridge(random_state=42), ALPHA_GRID, cv=kf, scoring="r2", n_jobs=-1
    )
    ridge_search.fit(X_train_scaled, y_train)
    lasso_search = GridSearchCV(
        Lasso(max_iter=50000, random_state=42),
        ALPHA_GRID,
        cv=kf,
        scoring="r2",
        n_jobs=-1,
    )
    lasso_search.fit(X_train_scaled, y_train)

    best_ridge = float(ridge_search.best_params_["alpha"])
    best_lasso = float(lasso_search.best_params_["alpha"])
    print(f"Лучший alpha Ridge: {best_ridge} (CV R² = {ridge_search.best_score_:.6f})")
    print(f"Лучший alpha Lasso: {best_lasso} (CV R² = {lasso_search.best_score_:.6f})")

    ridge_cv = pd.DataFrame(
        {
            "alpha": ALPHA_GRID["alpha"],
            "Ridge CV R²": ridge_search.cv_results_["mean_test_score"],
        }
    )
    lasso_cv = pd.DataFrame(
        {
            "alpha": ALPHA_GRID["alpha"],
            "Lasso CV R²": lasso_search.cv_results_["mean_test_score"],
        }
    )
    cv_table = ridge_cv.merge(lasso_cv, on="alpha")
    cv_table.to_csv(ROOT / "laba1_alpha_cv.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(ridge_cv["alpha"], ridge_cv["Ridge CV R²"], "o-", label="Ridge")
    ax.semilogx(lasso_cv["alpha"], lasso_cv["Lasso CV R²"], "s-", label="Lasso")
    ax.axvline(best_ridge, color="C0", linestyle="--", alpha=0.7, label=f"Ridge={best_ridge:g}")
    ax.axvline(best_lasso, color="C1", linestyle="--", alpha=0.7, label=f"Lasso={best_lasso:g}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Средний R² на CV")
    ax.set_title("Подбор силы регуляризации")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_alpha_gridsearch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/06_alpha_gridsearch.png")
    return best_ridge, best_lasso, ridge_search, lasso_search


def fit_original_models(
    X_train_scaled, X_test_scaled, y_train, y_test, kf, best_ridge, best_lasso, feature_names
):
    model_lin = LinearRegression()
    model_ridge = Ridge(alpha=best_ridge, random_state=42)
    model_lasso = Lasso(alpha=best_lasso, max_iter=50000, random_state=42)
    models = {
        "Линейная (OLS)": model_lin,
        f"Ridge (alpha={best_ridge:g})": model_ridge,
        f"Lasso (alpha={best_lasso:g})": model_lasso,
    }

    results = {}
    preds = {}
    print("=== Модели на исходных признаках ===")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="r2")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        preds[name] = y_pred
        rmse, r2, mape = regression_metrics(y_test, y_pred)
        results[name] = {
            "R2_CV_mean": float(cv_scores.mean()),
            "R2_CV_std": float(cv_scores.std()),
            "R2_Test": r2,
            "RMSE_Test": rmse,
            "MAPE_Test": mape / 100.0,
        }
        print(
            f"{name:<28} | R²(CV): {cv_scores.mean():.4f} | "
            f"R²(тест): {r2:.4f} | RMSE: {rmse:.2f} $ | MAPE: {mape:.2f}%"
        )

    coef_df = pd.DataFrame(
        {
            "Признак": feature_names,
            "Линейная": model_lin.coef_,
            "Ridge": model_ridge.coef_,
            "Lasso": model_lasso.coef_,
        }
    )
    coef_df.to_csv(ROOT / "laba1_weights.csv", index=False, encoding="utf-8-sig")
    print(f"Intercept OLS: {model_lin.intercept_:.2f} $")

    # прогнозы и остатки
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (name, _) in enumerate(models.items()):
        y_pred = preds[name]
        ax = axes[0, i]
        ax.scatter(
            y_test, y_pred, alpha=0.35, s=14, color="#2b5c8f", edgecolor="white", linewidth=0.2
        )
        low = min(float(y_test.min()), float(y_pred.min()))
        high = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([low, high], [low, high], "r--", label="прогноз = факт")
        ax.set_title(name)
        ax.set_xlabel("Факт, $")
        ax.set_ylabel("Прогноз, $")
        ax.legend(fontsize=8)

        ax_r = axes[1, i]
        residuals = y_test.values - y_pred
        ax_r.scatter(
            y_pred, residuals, alpha=0.35, s=14, color="#d62728", edgecolor="white", linewidth=0.2
        )
        ax_r.axhline(0.0, color="black", linestyle="--")
        ax_r.set_title(f"Остатки: {name}")
        ax_r.set_xlabel("Прогноз, $")
        ax_r.set_ylabel("Остаток, $")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_predictions_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/07_predictions_residuals.png")
    return models, results, preds, coef_df


def run_pca(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    kf,
    feature_names,
    best_ridge,
    best_lasso,
):
    pca_full = PCA().fit(X_train_scaled)
    eigenvalues = pca_full.explained_variance_
    exp_var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_ratio)
    k_kaiser = int(np.sum(eigenvalues > 1))
    k_var85 = int(np.argmax(cum_var >= 0.85) + 1)
    print(f"Кайзер: {k_kaiser} ГК ({cum_var[k_kaiser-1]:.4f})")
    print(f"Порог 85%: {k_var85} ГК ({cum_var[k_var85-1]:.4f})")

    pca_table = pd.DataFrame(
        {
            "Главная компонента": [f"ГК {i}" for i in range(1, len(eigenvalues) + 1)],
            "Собственное значение": eigenvalues,
            "Доля дисперсии": exp_var_ratio,
            "Накопленная доля": cum_var,
        }
    )
    pca_table.to_csv(ROOT / "laba1_pca_variance.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-", linewidth=2, markersize=7)
    axes[0].axhline(1.0, color="red", linestyle="--", linewidth=2, label="Кайзер (λ = 1)")
    axes[0].set_title("График каменистой осыпи (Scree Plot)")
    axes[0].set_xlabel("Номер главной компоненты")
    axes[0].set_ylabel("Собственное значение")
    axes[0].set_xticks(range(1, len(eigenvalues) + 1))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, len(cum_var) + 1), cum_var, "ro-", linewidth=2, markersize=7)
    axes[1].axhline(0.85, color="orange", linestyle="--", label="Порог 85%")
    axes[1].axvline(k_kaiser, color="green", linestyle=":", label=f"Кайзер: {k_kaiser}")
    axes[1].axvline(k_var85, color="gray", linestyle=":", label=f"85%: {k_var85}")
    axes[1].set_title("Кумулятивная объяснённая дисперсия")
    axes[1].set_xlabel("Число главных компонент")
    axes[1].set_ylabel("Доля дисперсии")
    axes[1].set_xticks(range(1, len(cum_var) + 1))
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_pca_scree.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/08_pca_scree.png")

    pca_for_labels = PCA(n_components=k_var85).fit(X_train_scaled)
    loadings = pd.DataFrame(
        pca_for_labels.components_.T,
        index=feature_names,
        columns=[f"ГК {i}" for i in range(1, k_var85 + 1)],
    )
    loadings.to_csv(ROOT / "laba1_pca_loadings.csv", encoding="utf-8-sig")

    pc_labels = {}
    for i in range(1, k_var85 + 1):
        title, detail = label_component(loadings[f"ГК {i}"])
        pc_labels[i] = title
        print(f"ГК {i} — «{title}»: {detail}")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(
        loadings, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax
    )
    ax.set_title(f"Факторные нагрузки первых {k_var85} главных компонент")
    ax.set_ylabel("Исходные признаки")
    ax.set_xlabel("Главные компоненты")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_pca_loadings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/09_pca_loadings.png")

    results_pca = {}
    for k in sorted({k_kaiser, k_var85}):
        pca_k = PCA(n_components=k).fit(X_train_scaled)
        X_train_pca = pca_k.transform(X_train_scaled)
        X_test_pca = pca_k.transform(X_test_scaled)
        models_k = {
            f"Линейная (PCA, {k} ГК)": LinearRegression(),
            f"Ridge (PCA, {k} ГК)": Ridge(alpha=best_ridge, random_state=42),
            f"Lasso (PCA, {k} ГК)": Lasso(
                alpha=best_lasso, max_iter=50000, random_state=42
            ),
        }
        results_pca[k] = {}
        rule = "Кайзер" if k == k_kaiser else "порог 85%"
        print(f"\n=== Модели на {k} ГК ({rule}) ===")
        for name, model in models_k.items():
            cv_scores = cross_val_score(model, X_train_pca, y_train, cv=kf, scoring="r2")
            model.fit(X_train_pca, y_train)
            y_pred = model.predict(X_test_pca)
            rmse, r2, mape = regression_metrics(y_test, y_pred)
            results_pca[k][name] = {
                "R2_CV_mean": float(cv_scores.mean()),
                "R2_CV_std": float(cv_scores.std()),
                "R2_Test": r2,
                "RMSE_Test": rmse,
                "MAPE_Test": mape / 100.0,
            }
            print(
                f"{name:<28} | R²(CV): {cv_scores.mean():.4f} | "
                f"R²(тест): {r2:.4f} | RMSE: {rmse:.2f} $ | MAPE: {mape:.2f}%"
            )

    return (
        k_kaiser,
        k_var85,
        eigenvalues,
        exp_var_ratio,
        cum_var,
        loadings,
        pc_labels,
        results_pca,
    )


def compare_and_save(
    results_orig,
    results_pca,
    k_kaiser,
    k_var85,
    feature_names,
    models_orig,
    best_ridge,
    best_lasso,
    ridge_search,
    lasso_search,
    eigenvalues,
    pc_labels,
    loadings,
    X_train_scaled,
    X_test_scaled,
):
    summary_rows = []
    for name, res in results_orig.items():
        summary_rows.append(
            {
                "Модель": name,
                "Признаки": f"Исходные ({len(feature_names)})",
                "R2 (CV)": res["R2_CV_mean"],
                "R2 (тест)": res["R2_Test"],
                "RMSE": res["RMSE_Test"],
                "MAPE, %": res["MAPE_Test"] * 100,
            }
        )
    for k, res_dict in results_pca.items():
        space = f"PCA ({k} ГК, {'Кайзер' if k == k_kaiser else '85%'})"
        for name, res in res_dict.items():
            summary_rows.append(
                {
                    "Модель": name,
                    "Признаки": space,
                    "R2 (CV)": res["R2_CV_mean"],
                    "R2 (тест)": res["R2_Test"],
                    "RMSE": res["RMSE_Test"],
                    "MAPE, %": res["MAPE_Test"] * 100,
                }
            )
    summary = (
        pd.DataFrame(summary_rows)
        .sort_values("R2 (тест)", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(ROOT / "laba1_metrics_summary.csv", index=False, encoding="utf-8-sig")
    print("\nСводная таблица метрик:")
    print(summary.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.barplot(
        data=summary, x="Модель", y="R2 (тест)", hue="Признаки", ax=axes[0], edgecolor="black"
    )
    axes[0].set_title("R² на тесте")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(
        data=summary, x="Модель", y="RMSE", hue="Признаки", ax=axes[1], edgecolor="black"
    )
    axes[1].set_title("RMSE, $")
    axes[1].tick_params(axis="x", rotation=30)
    sns.barplot(
        data=summary, x="Модель", y="MAPE, %", hue="Признаки", ax=axes[2], edgecolor="black"
    )
    axes[2].set_title("MAPE, %")
    axes[2].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "10_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/10_metrics_comparison.png")

    weights_payload = {
        "датасет": "camnugent/california-housing-prices",
        "обучающая_выборка": int(X_train_scaled.shape[0]),
        "тестовая_выборка": int(X_test_scaled.shape[0]),
        "подбор_гиперпараметров": {
            "сетка_alpha": ALPHA_GRID["alpha"],
            "лучший_alpha_Ridge": best_ridge,
            "лучший_alpha_Lasso": best_lasso,
            "CV_R2_Ridge": float(ridge_search.best_score_),
            "CV_R2_Lasso": float(lasso_search.best_score_),
        },
        "модели_на_исходных_признаках": {
            name: {
                **res,
                "intercept": float(models_orig[name].intercept_),
                "coefficients": {
                    col: float(coef)
                    for col, coef in zip(feature_names, models_orig[name].coef_)
                },
            }
            for name, res in results_orig.items()
        },
        "факторный_анализ_pca": {
            "компонент_по_Кайзеру": int(k_kaiser),
            "компонент_по_порогу_85": int(k_var85),
            "подписи_компонент_из_нагрузок": {f"ГК_{i}": pc_labels[i] for i in pc_labels},
            "собственные_значения": [float(e) for e in eigenvalues],
            "факторные_нагрузки": {
                col: {
                    f"ГК_{i}": float(loadings.loc[col, f"ГК {i}"])
                    for i in range(1, k_var85 + 1)
                }
                for col in feature_names
            },
        },
        "модели_на_главных_компонентах": {
            f"{k}_ГК": results_pca[k] for k in results_pca
        },
    }
    with open(ROOT / "laba1_model_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, ensure_ascii=False, indent=2)
    print("Сохранены: laba1_model_weights.json, laba1_weights.csv, laba1_metrics_summary.csv")


def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    print("=== laba1_models.py: регрессия и PCA ===\n")
    df_model = load_prepared()
    (
        feature_names,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        kf,
        _,
    ) = split_and_scale(df_model)

    best_ridge, best_lasso, ridge_search, lasso_search = tune_regularization(
        X_train_scaled, y_train, kf
    )
    models_orig, results_orig, _, _ = fit_original_models(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        kf,
        best_ridge,
        best_lasso,
        feature_names,
    )
    (
        k_kaiser,
        k_var85,
        eigenvalues,
        _,
        _,
        loadings,
        pc_labels,
        results_pca,
    ) = run_pca(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        kf,
        feature_names,
        best_ridge,
        best_lasso,
    )
    compare_and_save(
        results_orig,
        results_pca,
        k_kaiser,
        k_var85,
        feature_names,
        models_orig,
        best_ridge,
        best_lasso,
        ridge_search,
        lasso_search,
        eigenvalues,
        pc_labels,
        loadings,
        X_train_scaled,
        X_test_scaled,
    )
    print("\nГотово. Все графики в папке figures/")


if __name__ == "__main__":
    main()
