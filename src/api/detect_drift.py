import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

def detect_drift_ks(train_column, prod_column, threshold=0.05):
    """
    Обнаруживает дрифт для непрерывного признака с помощью теста Колмогорова-Смирнова.
    H0: Выборки взяты из одного и того же распределения.
    p-value < threshold -> отвергаем H0, есть дрифт.
    """
    statistic, p_value = ks_2samp(train_column, prod_column)
    return p_value < threshold, p_value


def detect_drift_chisquare(train_column, prod_column, threshold=0.05):
    """
    Обнаруживает дрифт для категориального признака с помощью теста хи-квадрат.
    """
    # Создаем таблицы сопряженности
    train_counts = train_column.value_counts().sort_index()
    prod_counts = prod_column.value_counts().sort_index()

    # Выравниваем индексы, на случай если в одной выборке нет каких-то категорий
    all_categories = train_counts.index.union(prod_counts.index)
    train_counts = train_counts.reindex(all_categories, fill_value=0)
    prod_counts = prod_counts.reindex(all_categories, fill_value=0)

    contingency_table = np.array([train_counts, prod_counts])

    statistic, p_value, dof, expected = chi2_contingency(contingency_table)
    return p_value < threshold, p_value