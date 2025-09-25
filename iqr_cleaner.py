from sklearn import datasets
import pandas as pd


def get_iqr_bounds(dataframe, column, k=1.5):
    """Вычисляет нижнюю и верхнюю границы для выбросов по IQR."""
    if column not in dataframe.columns:
        raise ValueError(f"Колонка '{column}' отсутствует в DataFrame")

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise TypeError(f"Колонка '{column}' должна содержать числовые данные")

    q1, q3 = dataframe[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return lower_bound, upper_bound


def clean_by_iqr(dataframe, column, k=1.5, fillna_strategy="drop"):
    """Удаляет выбросы по колонке с использованием IQR. Также обрабатывает NaN (drop или fill)."""
    df = dataframe.copy()

    # Обработка NaN
    if fillna_strategy == "drop":
        df = df.dropna(subset=[column])
    elif fillna_strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif fillna_strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    else:
        raise ValueError("fillna_strategy должен быть 'drop', 'median' или 'mean'")

    lower_bound, upper_bound = get_iqr_bounds(df, column, k)
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return cleaned_df


def main():
    """Основной скрипт: загрузка данных, очистка и вывод результатов."""
    prices = datasets.fetch_california_housing(as_frame=True)
    data_pd = prices.data

    data_cleaned = clean_by_iqr(data_pd, "MedInc", k=1.5, fillna_strategy="drop")

    print("Количество удалённых строк:", len(data_pd) - len(data_cleaned))


if __name__ == "__main__":
    main()
