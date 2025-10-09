from z_score import ZScoreOutlierDetector
from sklearn.datasets import load_diabetes
import pandas as pd


def detect_outliers(df):
    """Обнаруживает выбросы с помощью ZScoreOutlierDetector"""
    detector = ZScoreOutlierDetector()
    detector.fit(df)
    outlier_indices = detector.get_outliers_indices()

    df = df.copy()
    df["flag"] = df.index.isin(outlier_indices).astype(int)
    return df


def analyze_outliers(df):
    """Анализирует частоту выбросов в DataFrame"""
    count = df["flag"].value_counts()
    ratio = df["flag"].value_counts(normalize=True) * 100

    print("\nАнализ выбросов")
    print(f"Количество:\n{count}")
    print(f"\nДоля (%):\n{ratio.round(2)}")


def main():
    diabetes = load_diabetes()
    data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # 1. Поиск выбросов
    data_with_flags = detect_outliers(data)

    # 2. Анализ частоты
    analyze_outliers(data_with_flags)

    print("\nПример данных с флагом выброса:")
    print(data_with_flags.head())


if __name__ == "__main__":
    main()
