import pandas as pd
import numpy as np

def create_synthetic_df():
    """Создаёт датафрейм с пропусками и целевой переменной."""
    df = pd.DataFrame({
        "feature": [10, np.nan, 15, np.nan, 20, 25, np.nan],
        "target":   [0, 1, 0, 1, 0, 1, 1]
    })
    return df

def add_missing_indicator(df, column):
    """
    Добавляет бинарный индикатор пропусков для указанного столбца.
    1 — если значение NaN, 0 — если есть значение.
    """
    new_col = f"{column}_missing"
    df[new_col] = df[column].isna().astype(int)
    return df

def analyze_df(df, missing_col, target_col):
    """
    Анализирует, связаны ли пропуски с целевой переменной.
    Возвращает среднее значение target по группам (0/1) наличия пропуска.
    """
    return df.groupby(missing_col)[target_col].mean()

def main():
    df = create_synthetic_df()
    print("Исходные данные:\n", df, "\n")

    df = add_missing_indicator(df, "feature")
    print("С добавленным признаком:\n", df, "\n")

    relation = analyze_df(df, "feature_missing", "target")
    print("Связь пропусков с целевой переменной:\n", relation)

if __name__ == "__main__":
    main()


