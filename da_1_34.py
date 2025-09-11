from sklearn import datasets # для датасета


def clean_by_iqr(dataframe, col): # функция для вычисления и удаления выбросов по столбцу
    q1, q3 = dataframe[col].quantile([0.25, 0.75])
    IQR = q3-q1
    bottom_border = q1-1.5*IQR # устанавливаем границы
    upper_bottom = q3+1.5*IQR

    data_cleaned = dataframe[(dataframe[col] >= bottom_border) & (dataframe[col] <= upper_bottom)] # выкидываем выбросы

    return data_cleaned


prices = datasets.fetch_california_housing(as_frame=True) # загружаем датасет
data_pd = prices.data
data_cleaned = clean_by_iqr(data_pd, "MedInc")

print(data_pd["MedInc"].size - data_cleaned["MedInc"].size) # output - 681 - сравниваем количество строк
