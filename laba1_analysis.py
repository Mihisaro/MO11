# -*- coding: utf-8 -*-
"""
Лабораторная работа №1 — часть 1: анализ данных California Housing Prices.

Загрузка датасета, предобработка, описательная статистика, распределения,
зависимости от цены, корреляции и VIF. Графики сохраняются в папку figures/.
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
    if local_csv.exists():
        df_raw = pd.read_csv(local_csv)
        print(f"Загружен локальный файл: {local_csv}")
    else:
        path = kagglehub.dataset_download("camnugent/california-housing-prices")
        csv_path = next(Path(path).rglob("*.csv"))
        print(f"Скачано через kagglehub: {csv_path}")
        df_raw = pd.read_csv(csv_path)
        df_raw.to_csv(local_csv, index=False)

    df = df_raw.rename(columns=COL_MAP)
    print(f"Размерность: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print("Пропуски:\n", df.isnull().sum())

    n_miss = int(df["Всего спален"].isnull().sum())
    if n_miss:
        med = float(df["Всего спален"].median())
        df["Всего спален"] = df["Всего спален"].fillna(med)
        print(f"Заполнено пропусков «Всего спален»: {n_miss}, медиана = {med:.1f}")

    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Удалено дубликатов: {dups}")
    else:
        print("Дубликаты не обнаружены.")

    dummies = pd.get_dummies(df[CAT], prefix="Океан", drop_first=True, dtype=float)
    df_model = pd.concat([df.drop(columns=[CAT]), dummies], axis=1)
    print(f"После кодирования: {df_model.shape}")
    return df, df_model


def descriptive_stats(df_model: pd.DataFrame) -> pd.DataFrame:
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
    print("Описательная статистика сохранена: laba1_descriptive_stats.csv")
    return out


def plot_distributions(df: pd.DataFrame, df_model: pd.DataFrame) -> None:
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
    print("Сохранён график: figures/01_distributions.png")

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
    corr_matrix = df_model.corr(numeric_only=True)

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
    print("Корреляция с ценой:\n", corr_target)

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
    print("VIF:\n", vif_df.round(3))

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
    plot_scatter_vs_target(df_model)
    correlation_and_vif(df_model)
    df_model.to_csv(ROOT / "housing_prepared.csv", index=False, encoding="utf-8-sig")
    print("\nПодготовленные данные: housing_prepared.csv")
    print("Готово. Графики в папке figures/")


if __name__ == "__main__":
    main()
