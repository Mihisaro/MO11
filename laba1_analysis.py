# -*- coding: utf-8 -*-
"""
Лабораторная работа №1 — часть 1: анализ данных California Housing Prices.

Загрузка датасета, предобработка, описательная статистика, распределения,
зависимости от цены, корреляции и VIF. Графики сохраняются в папку figures/.

Номера вопросов защиты (искать в коде по «#1.», «#12.» — с точкой):
  #1 загрузка CSV и параметры read_csv
  #2 число наблюдений, признаков и типы столбцов
  #3 имя целевой переменной (сами X и y — в laba1_models.py)
  #4 какие столбцы исключены из признаков
  #5 пропуски и дубликаты
  #6 описательные статистики
  #7 распределения
  #8 выбросы
  #9 обработка пропусков и сколько строк осталось
  #10 категориальное кодирование
  #14 матрица корреляций
  #15 метод корреляции
  #16 самые сильные пары
  #17 VIF
  #18 что делать с мультиколлинеарными признаками
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

TARGET = "Медианная цена дома"
CAT = "Близость к океану"

COL_MAP = {
    "longitude": "Долгота",
    "latitude": "Широта",
    "housing_median_age": "Медианный возраст дома",
    "total_rooms": "Всего комнат",
    "total_bedrooms": "Всего спален",
    "population": "Население",
    "households": "Домохозяйства",
    "median_income": "Медианный доход",
    "median_house_value": "Медианная цена дома",
    "ocean_proximity": "Близость к океану",
}


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузка с Kaggle, заполнение пропусков, one-hot кодирование."""
    local_csv = ROOT / "housing.csv"
    # #1. Загрузка. Дополнительные аргументы read_csv не заданы:
    # sep=',' и header=0 берутся по умолчанию. Это обычный CSV Kaggle
    # с заголовком в первой строке, поэтому sep и encoding указывать не нужно.
    if local_csv.exists():
        df_raw = pd.read_csv(local_csv)
        print(f"#1. Загружен локальный файл: {local_csv}")
    else:
        path = kagglehub.dataset_download("camnugent/california-housing-prices")
        csv_path = next(Path(path).rglob("*.csv"))
        print(f"#1. Скачано через kagglehub: {csv_path}")
        df_raw = pd.read_csv(csv_path)
        df_raw.to_csv(local_csv, index=False)

    df = df_raw.rename(columns=COL_MAP)
    # #2. Число наблюдений — строки, число столбцов — shape[1] (9 признаков + цель).
    # Типы: dtypes. object — категория, остальные столбцы числовые.
    n_obs, n_cols = df.shape
    print(f"#2. Наблюдений: {n_obs}, столбцов: {n_cols}")
    print("#2. Типы столбцов:\n", df.dtypes.to_string())

    # #3. Целевая переменная — TARGET («Медианная цена дома»).
    # Массивы X и y собираются позже, в laba1_models.split_and_scale.
    print(f"#3. Целевая переменная: {TARGET}")

    # #5. Пропуски — isnull().sum() по каждому столбцу.
    # Дубликаты — duplicated().sum() по полностью совпадающим строкам.
    print("#5. Пропуски:\n", df.isnull().sum().to_string())
    n_before = len(df)
    n_miss = int(df["Всего спален"].isnull().sum())
    n_if_drop = int(df.dropna().shape[0])
    # #9. Строки с пропусками не удаляются: 207 пропусков — около 1%.
    # Пропуск заполняется медианой столбца «Всего спален».
    # Если бы вызвали dropna(), осталось бы n_if_drop строк.
    if n_miss:
        med = float(df["Всего спален"].median())
        df["Всего спален"] = df["Всего спален"].fillna(med)
        print(
            f"#9. Заполнено пропусков «Всего спален»: {n_miss}, медиана = {med:.1f}. "
            f"Было строк: {n_before}, осталось: {len(df)}. "
            f"При удалении строк осталось бы {n_if_drop}."
        )
    else:
        print(f"#9. Пропусков нет, удалять нечего. Строк: {len(df)}.")

    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"#5. Удалено дубликатов: {dups}. Новый размер: {df.shape}")
    else:
        print("#5. Дубликаты не обнаружены.")

    # #10. Единственный категориальный признак — «Близость к океану»
    # (5 классов, порядка между ними нет). Поэтому one-hot, а не LabelEncoder:
    # числа 0,1,2,... модель приняла бы за порядок классов.
    # drop_first=True убирает базовый класс <1H OCEAN, иначе дамми линейно зависимы.
    print("#10. Классы категории:\n", df[CAT].value_counts().to_string())
    dummies = pd.get_dummies(df[CAT], prefix="Океан", drop_first=True, dtype=float)
    # #4. Из признаков убран исходный текстовый столбец категории: его заменяют дамми.
    # Числовые столбцы здесь не удаляются. Решение по VIF — метка #18.
    df_model = pd.concat([df.drop(columns=[CAT]), dummies], axis=1)
    print(f"#4. #10. После кодирования: {df_model.shape}. Столбцы: {list(df_model.columns)}")
    return df, df_model


def descriptive_stats(df_model: pd.DataFrame) -> pd.DataFrame:
    # #6. describe() даёт count, mean, std, min, квартили и max.
    # Дополнительно: медиана, мода, IQR = Q3 − Q1 и асимметрия skew().
    stats_df = df_model.describe().T
    stats_df["Медиана"] = df_model.median()
    stats_df["Мода"] = [df_model[c].mode().iloc[0] for c in df_model.columns]
    stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]
    stats_df["Асимметрия"] = df_model.skew()
    stats_df = stats_df.rename(
        columns={
            "count": "Количество",
            "mean": "Среднее",
            "std": "Станд. откл.",
            "min": "Мин.",
            "max": "Макс.",
        }
    )
    cols = [
        "Количество",
        "Среднее",
        "Станд. откл.",
        "Медиана",
        "Мода",
        "IQR",
        "Мин.",
        "Макс.",
        "Асимметрия",
    ]
    out = stats_df[cols].round(2)
    out.to_csv(ROOT / "laba1_descriptive_stats.csv", encoding="utf-8-sig")
    print("#6. Описательная статистика сохранена: laba1_descriptive_stats.csv")
    price = df_model[TARGET]
    print(
        "#6. Цена скошена вправо: среднее "
        f"{price.mean():.0f} > медиана {price.median():.0f}, "
        f"максимум {price.max():.0f} (потолок датасета). "
        "Масштабы столбцов разные — перед Ridge/Lasso/PCA нужна стандартизация."
    )
    return out


def plot_distributions(df: pd.DataFrame, df_model: pd.DataFrame) -> None:
    # #7. Распределения числовых признаков и цели — histplot.
    # Категория «Близость к океану» — countplot (частоты классов).
    sns.set_theme(style="whitegrid", palette="muted")
    plot_cols = [
        TARGET,
        "Медианный доход",
        "Медианный возраст дома",
        "Всего комнат",
        "Всего спален",
        "Население",
        "Домохозяйства",
        "Широта",
        "Долгота",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    colors = sns.color_palette("husl", len(plot_cols))
    for ax, col, color in zip(axes.ravel(), plot_cols, colors):
        sns.histplot(df_model[col], kde=True, ax=ax, color=color, bins=30, edgecolor="black")
        ax.axvline(df_model[col].mean(), color="red", linestyle="--", label="Среднее")
        ax.axvline(df_model[col].median(), color="green", linestyle="-", label="Медиана")
        ax.set_title(f"Распределение: {col}")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("#7. Сохранён график: figures/01_distributions.png")
    print(
        "#7. Цена и доход скошены вправо, у цены пик около 500000. "
        "Комнаты, спальни, население и домохозяйства имеют тяжёлый правый хвост. "
        "По категории больше всего <1H OCEAN и INLAND, класс ISLAND почти пустой."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    order = df[CAT].value_counts().index
    sns.countplot(data=df, x=CAT, order=order, ax=ax, edgecolor="black")
    ax.set_title("Распределение категории «Близость к океану»")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Число районов")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_ocean_proximity_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/02_ocean_proximity_counts.png")


def analyze_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # #8. Выбросы: ящик с усами и правило Тьюки (за пределами Q1 − 1.5·IQR и Q3 + 1.5·IQR).
    num_cols = [c for c in df.columns if c != CAT]
    rows = []
    for col in num_cols:
        q1 = float(df[col].quantile(0.25))
        q3 = float(df[col].quantile(0.75))
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append(
            {
                "Признак": col,
                "Нижняя граница": low,
                "Верхняя граница": high,
                "Число точек за границей": n_out,
                "Доля, %": round(100.0 * n_out / len(df), 2),
            }
        )
    out = pd.DataFrame(rows)
    print("#8. Точки за границами 1.5·IQR:\n", out.to_string(index=False))
    print(
        "#8. Решение: строки не удаляем. Хвосты — крупные районы, а не ошибки ввода. "
        "Верх цены 500001 — цензурирование исходного датасета, его тоже оставляем."
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.boxplot(x=df[col], ax=ax, color="#9ecae1")
        ax.set_title(col)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02b_outliers_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("#8. Сохранён график: figures/02b_outliers_boxplots.png")
    return out


def plot_scatter_vs_target(df_model: pd.DataFrame) -> None:
    features = [
        "Медианный доход",
        "Всего комнат",
        "Всего спален",
        "Население",
        "Домохозяйства",
        "Медианный возраст дома",
        "Широта",
        "Долгота",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, col in zip(axes.ravel(), features):
        sns.scatterplot(
            data=df_model, x=col, y=TARGET, ax=ax, alpha=0.25, s=12, color="#2b5c8f"
        )
        sns.regplot(
            data=df_model, x=col, y=TARGET, scatter=False, ax=ax, color="red", ci=None
        )
        ax.set_title(f"Цена vs {col}")
        ax.set_ylabel("Цена, $")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_scatter_vs_price.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/03_scatter_vs_price.png")


def correlation_and_vif(df_model: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # #14. В матрицу входят все столбцы df_model: числовые признаки, дамми океана и цель.
    # #15. Метод — Пирсон (значение method по умолчанию у DataFrame.corr).
    # Он измеряет линейную связь, а модели ниже тоже линейные.
    # Спирмен измерял бы ранги и к коэффициентам линейной регрессии не привязан.
    corr_matrix = df_model.corr(method="pearson", numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Матрица корреляций Пирсона")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/04_correlation_matrix.png")

    corr_target = (
        corr_matrix[TARGET]
        .drop(TARGET)
        .sort_values(ascending=False)
        .round(3)
        .to_frame("Корреляция с ценой")
    )
    corr_target.to_csv(ROOT / "laba1_corr_with_target.csv", encoding="utf-8-sig")
    print("#14. #15. Корреляция Пирсона с ценой:\n", corr_target)

    # #16. Сила связи между парой — |r| вне диагонали. Верхний треугольник без повторов.
    pair_abs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        .stack()
        .abs()
        .sort_values(ascending=False)
    )
    print("#16. Самые сильные пары по |r|:\n", pair_abs.head(8).round(3).to_string())

    # #17. В VIF передаются только признаки, без целевой цены.
    # Перед расчётом столбцы стандартизируются: на VIF это не влияет,
    # но ставит признаки в один масштаб с дальнейшими моделями.
    X_all = df_model.drop(columns=[TARGET])
    X_vif = pd.DataFrame(
        StandardScaler().fit_transform(X_all), columns=X_all.columns
    )
    vif_df = pd.DataFrame(
        {
            "Признак": X_vif.columns,
            "VIF": [
                variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])
            ],
        }
    ).sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_df.to_csv(ROOT / "laba1_vif.csv", index=False, encoding="utf-8-sig")
    print("#17. VIF:\n", vif_df.round(3).to_string(index=False))
    print(
        "#17. Ориентир: VIF < 5 — коллинеарности нет, 5–10 — умеренная, > 10 — сильная. "
        "Большой VIF значит, что признак почти линейно выражается через остальные."
    )
    # #18. Столбцы с VIF > 10 не удаляем и не склеиваем вручную:
    # вместе они описывают размер района и географию.
    # Дальше мультиколлинеарность гасится регуляризацией (Ridge/Lasso) и PCA.
    high_vif = vif_df.loc[vif_df["VIF"] > 10, "Признак"].tolist()
    print(f"#18. Не удаляем, несмотря на VIF > 10: {high_vif}")
    print("#18. Вместо удаления — Ridge/Lasso и PCA в laba1_models.py.")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(vif_df["Признак"], vif_df["VIF"], color="#3182bd", edgecolor="black")
    ax.axvline(5.0, color="orange", linestyle="--", linewidth=1.5, label="VIF = 5")
    ax.axvline(10.0, color="red", linestyle="--", linewidth=1.5, label="VIF = 10")
    ax.set_title("Коэффициенты VIF (мультиколлинеарность)")
    ax.set_xlabel("VIF")
    for bar in bars:
        val = bar.get_width()
        ax.text(
            val + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
        )
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_vif.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Сохранён график: figures/05_vif.png")
    return corr_target, vif_df


def main() -> None:
    print("=== laba1_analysis.py: анализ California Housing Prices ===\n")
    df, df_model = load_and_prepare()
    print("\n", descriptive_stats(df_model), "\n")
    plot_distributions(df, df_model)
    analyze_outliers(df)
    plot_scatter_vs_target(df_model)
    correlation_and_vif(df_model)
    df_model.to_csv(ROOT / "housing_prepared.csv", index=False, encoding="utf-8-sig")
    print("\nПодготовленные данные: housing_prepared.csv")
    print("Готово. Графики в папке figures/")


if __name__ == "__main__":
    main()
