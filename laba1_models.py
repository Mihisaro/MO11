# -*- coding: utf-8 -*-
"""
Лабораторная работа №1 — часть 2: регрессия и PCA.

Train/test, стандартизация, OLS / Ridge / Lasso (GridSearchCV),
графики прогнозов и остатков, PCA (каменистая осыпь), сравнение метрик.
Графики сохраняются в figures/, веса — в csv/json.

Номера вопросов защиты (искать в коде по «#19.», «#36.» — с точкой):
  #3  формирование X и y
  #11 какие признаки масштабируются
  #12 fit, transform и fit_transform
  #13 на каких данных учатся преобразования
  #19 разбиение train/test и размеры
  #20 зачем random_state
  #21 подходит ли случайное разбиение
  #22 где создаются OLS, Lasso и Ridge
  #23 явные параметры и значения по умолчанию
  #24 подбор alpha
  #25 кросс-валидация: число фолдов и метрика
  #26 масштабирование и PCA внутри кросс-валидации
  #27 Pipeline
  #28 коэффициенты моделей
  #29 прогнозы и на какой выборке метрики
  #30 #31 #32 RMSE, R², MAPE и единицы
  #33 нули в целевой переменной и MAPE
  #34 train против test, переобучение
  #35 как выбирается лучшая модель
  #36 стандартизация перед PCA
  #37 где обучается PCA
  #38 как выбирается число компонент
  #39 explained_variance_ratio_ и накопленная доля
  #40 размеры матриц после PCA
  #41 преобразование тестовой выборки
  #42 нагрузки компонент
  #43 проверка мультиколлинеарности после PCA
  #44 повторное обучение моделей на компонентах
  #45 одинаковые ли тест и правила оценки до и после PCA
  #46 итоговая таблица: где PCA помог, а где нет
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

TARGET = "Медианная цена дома"
ALPHA_GRID = {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]}


def regression_metrics(y_true, y_pred):
    # #30. Сюда передаются фактические цены и прогнозы одной и той же выборки.
    # #31. RMSE считается из MSE: корень возвращает ошибку в тех же единицах, что и y.
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    # #30. R² — доля объяснённой дисперсии цены.
    r2 = float(r2_score(y_true, y_pred))
    # #32. sklearn считает MAPE как долю (0.29 = 29%). Умножение на 100 переводит в проценты.
    # RMSE остаётся в долларах, потому что y — медианная цена в долларах.
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
    # #3. X — все столбцы, кроме цели. y — столбец «Медианная цена дома».
    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET]
    feature_names = list(X.columns)

    # #33. MAPE делит ошибку на фактическую цену. Нулей и околонулевых цен нет,
    # минимум далеко от 0, поэтому метрика не раздувается из-за деления.
    n_zero = int((y == 0).sum())
    n_near = int((y.abs() < 1000).sum())
    print(
        f"#33. Нулей в цене: {n_zero}, значений |y| < 1000: {n_near}, "
        f"минимум = {float(y.min()):.0f}. MAPE можно читать как обычный процент."
    )

    # #19. 80/20. shuffle=True по умолчанию у train_test_split.
    # #20. random_state=42 фиксирует перемешивание: повторный запуск даёт те же строки.
    # Другое число перетасует выборки заново, размеры останутся теми же, метрики чуть сдвинутся.
    # #21. Случайное разбиение подходит: это одна перепись 1990 года, столбца даты нет,
    # повторных наблюдений одного объекта нет (строка = блок-группа, полных дубликатов нет, см. #5).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(
        f"#19. Train: {X_train.shape[0]} x {X_train.shape[1]}, "
        f"Test: {X_test.shape[0]} x {X_test.shape[1]}"
    )
    print(f"#21. Столбцы признаков: {feature_names}. Столбца времени среди них нет.")

    # #11. Масштабируются все столбцы X. Цена y не масштабируется:
    # RMSE тогда остаётся в долларах.
    # #12. fit учит среднее и СКО. transform применяет уже выученные параметры.
    # fit_transform(X_train) — это fit и transform на одних и тех же данных, результат тот же.
    # #13. fit вызывается только на X_train. К X_test — только transform,
    # поэтому тестовые среднее и СКО в настройку не попадают.
    # #36. Этот же scaler стоит перед PCA: PCA тоже получает X_train_scaled.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)
    # #25. 5 фолдов, перемешивание фиксировано тем же random_state.
    # Метрика кросс-валидации — R² (scoring="r2" в GridSearchCV и cross_val_score).
    # #26. Scaler уже обучен на всём train. Внутри фолдов он заново не учится.
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(
        "#12. Среднее стандартизованного train (ожидается ≈ 0):\n",
        X_train_scaled.mean().round(6).to_string(),
    )
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
    # #24. alpha перебирается по сетке. Для каждого значения — 5-fold CV, метрика R².
    # В перебор входит только обучающая выборка. Тест здесь не используется.
    print(f"#24. Проверяемые alpha: {ALPHA_GRID['alpha']}")
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
    # #22. Три модели создаются здесь и ниже обучаются на X_train_scaled.
    # #23. LinearRegression() — все параметры по умолчанию (fit_intercept=True).
    # Ridge: явно alpha и random_state; solver, fit_intercept и прочее — по умолчанию.
    # Lasso: явно alpha, max_iter=50000 и random_state; остальное по умолчанию.
    # random_state у Ridge/Lasso влияет только на случайный solver, не на данные.
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
        # #25. cross_val_score: 5 разбиений kf, метрика R², только на train.
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="r2")
        model.fit(X_train_scaled, y_train)
        # #29. Итоговые RMSE, R² и MAPE считаются на тесте: y_test и predict(X_test_scaled).
        y_pred = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)
        preds[name] = y_pred
        rmse, r2, mape = regression_metrics(y_test, y_pred)
        rmse_tr, r2_tr, mape_tr = regression_metrics(y_train, y_pred_train)
        results[name] = {
            "R2_CV_mean": float(cv_scores.mean()),
            "R2_CV_std": float(cv_scores.std()),
            "R2_Train": r2_tr,
            "R2_Test": r2,
            "RMSE_Train": rmse_tr,
            "RMSE_Test": rmse,
            "MAPE_Train": mape_tr / 100.0,
            "MAPE_Test": mape / 100.0,
        }
        print(
            f"{name:<28} | R²(CV): {cv_scores.mean():.4f} | "
            f"R²(тест): {r2:.4f} | RMSE: {rmse:.2f} $ | MAPE: {mape:.2f}%"
        )
        # #34. Сравнение train и test. Близкие R² и RMSE — сильного переобучения нет.
        print(
            f"#34. {name}: R² train={r2_tr:.4f} / test={r2:.4f}; "
            f"RMSE train={rmse_tr:.2f} / test={rmse:.2f} $"
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
    # #28. Коэффициенты при стандартизованных признаках: на сколько долларов
    # меняется прогноз, если признак вырос на 1 стандартное отклонение.
    print("#28. Коэффициенты:\n", coef_df.round(2).to_string(index=False))
    print(
        "#28. OLS, Ridge и Lasso дают близкие веса и одинаковые знаки. "
        "Lasso чуть сильнее сжимает самые слабые коэффициенты, до нуля их не обнуляет."
    )
    print(f"#28. Свободный член OLS: {model_lin.intercept_:.2f} $")

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
    # #37. PCA учится на стандартизованном train. В компонентах нет целевой цены.
    # #36. На вход уже подаётся X_train_scaled (scaler.fit был только на X_train).
    pca_full = PCA().fit(X_train_scaled)
    eigenvalues = pca_full.explained_variance_
    # #39. explained_variance_ratio_ — доля дисперсии каждой компоненты.
    # Накопленная доля — сумма этих долей с первой по текущую (cumsum).
    exp_var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_ratio)
    # #38. Число компонент не задано одной константой. Два правила:
    # Кайзер — оставляем компоненты с собственным значением > 1;
    # второе — минимум компонент, чтобы накопленная доля достигла 85%.
    k_kaiser = int(np.sum(eigenvalues > 1))
    k_var85 = int(np.argmax(cum_var >= 0.85) + 1)
    print("#39. explained_variance_ratio_:", np.round(exp_var_ratio, 4))
    print("#39. Накопленная доля:", np.round(cum_var, 4))
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

    # #42. Нагрузка — вес исходного признака в компоненте (строки components_).
    # Сильнее всего с компонентой связаны признаки с наибольшим |весом|.
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
        print(f"#42. ГК {i} — «{title}»: {detail}")

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
        # #37. Отдельный PCA на k компонент, снова только на train.
        pca_k = PCA(n_components=k).fit(X_train_scaled)
        X_train_pca = pca_k.transform(X_train_scaled)
        # #41. Тест переводится уже обученным PCA. fit на тесте не вызывается.
        X_test_pca = pca_k.transform(X_test_scaled)
        # #40. Число строк то же, число столбцов стало равно k.
        print(
            f"#40. До PCA: train {X_train_scaled.shape}, test {X_test_scaled.shape}. "
            f"После PCA: train {X_train_pca.shape}, test {X_test_pca.shape}"
        )
        # #43. Компоненты ортогональны: корреляции вне диагонали около 0, VIF около 1.
        pc_df = pd.DataFrame(X_train_pca, columns=[f"ГК {i}" for i in range(1, k + 1)])
        pc_corr = pc_df.corr()
        max_off = float(
            pc_corr.where(~np.eye(k, dtype=bool)).abs().max().max()
        )
        vif_pc = [
            float(variance_inflation_factor(pc_df.values, i)) for i in range(k)
        ]
        print(
            f"#43. Макс. |корреляция| между ГК: {max_off:.6f}. "
            f"VIF компонент: {[round(v, 3) for v in vif_pc]}"
        )
        # #44. Те же три регрессии, но уже на главных компонентах.
        # #45. Те же y_test, regression_metrics, alpha и random_state, что и до PCA.
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
            # #26. PCA уже обучен на всём train. Внутри фолдов заново не учится.
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
    # #35. Если метрики расходятся, приоритет такой: выше R² на тесте,
    # при равенстве — меньше RMSE, затем меньше MAPE.
    # На этих данных все три метрики смотрят в одну сторону.
    best_name = max(
        results_orig,
        key=lambda n: (
            results_orig[n]["R2_Test"],
            -results_orig[n]["RMSE_Test"],
            -results_orig[n]["MAPE_Test"],
        ),
    )
    print(f"#35. Лучшая модель на исходных признаках: {best_name}")
    # #46. Сводная таблица. PCA сравнивается с исходными признаками по тем же метрикам.
    print("\n#46. Сводная таблица метрик:")
    print(summary.round(4).to_string(index=False))
    orig_best = float(
        summary.loc[summary["Признаки"].str.startswith("Исходные"), "R2 (тест)"].max()
    )
    pca_best = float(
        summary.loc[summary["Признаки"].str.startswith("PCA"), "R2 (тест)"].max()
    )
    print(
        f"#46. Лучший R² без PCA: {orig_best:.4f}, лучший R² с PCA: {pca_best:.4f}. "
        "PCA ухудшил OLS, Ridge и Lasso: R² ниже, RMSE и MAPE выше."
    )

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
    # #27. sklearn.pipeline.Pipeline не используется. Шаги записаны явно, по порядку:
    # 1) train_test_split — тест отделяется до настройки преобразований;
    # 2) StandardScaler.fit только на train, transform на train и на test;
    # 3) GridSearchCV и cross_val_score — подбор alpha и R² на train;
    # 4) model.fit на всём train, итоговые метрики на test;
    # 5) PCA.fit на X_train_scaled, transform train и test, затем те же три модели.
    # Scaler и PCA снаружи кросс-валидации, внутри фолдов не переобучаются (#26).
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
