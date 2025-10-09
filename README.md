# Z-Score Outlier Detection

Этот проект демонстрирует использование метода **Z-score** для обнаружения выбросов в табличных данных.

## Описание
Скрипт:
1. Загружает данные (`load_diabetes` из `sklearn.datasets`).
2. Находит выбросы с помощью `ZScoreOutlierDetector`.
3. Добавляет бинарный признак `flag`:
   - `1` — выброс
   - `0` — нормальное значение
4. Анализирует частоту выбросов.

## Установка
```
git clone https://github.com/svetlana-fisher/AI-based-development.git
cd AI-based-development/z-score-counter
pip install -r requirements.txt
```

## Описание функций

#### `detect_outliers(df)`

Находит выбросы с помощью `ZScoreOutlierDetector` и добавляет столбец `flag`.

**Аргументы:**
- `df` (`pd.DataFrame`) — исходный датафрейм с числовыми признаками  

**Возвращает:**  
`pd.DataFrame` с добавленным бинарным столбцом `flag`, где:  
- `1` — строка является выбросом  
- `0` — строка нормальная  

---

#### `analyze_outliers(df)`

Анализирует частоту выбросов.

**Аргументы:**
- `df` (`pd.DataFrame`) — датафрейм с признаком `flag`  

**Вывод:**  
В консоль печатает:  
- количество строк с флагом `1` и `0`  
- процентное соотношение выбросов  

---

#### `main()`

Основная функция, которая:  
1. Загружает данные (`load_diabetes()` из `sklearn`)  
2. Вызывает `detect_outliers()`  
3. Вызывает `analyze_outliers()`  
4. Показывает пример датафрейма с флагами


