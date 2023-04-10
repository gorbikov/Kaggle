import pathlib

import pandas as pd
from sklearn.preprocessing import StandardScaler

from eda import *

# Вводные для удобства.
current_script_name = pathlib.Path(__file__).name
random_ceed = 777

# Загружает данные и создаёт датафреймы.
train_df_path = pathlib.Path("intermediate data/results/Part 1.py_train_df.csv")
test_df_path = pathlib.Path("intermediate data/results/Part 1.py_test_df.csv")

train_df = pd.read_csv(train_df_path)
test_df = pd.read_csv(test_df_path)

# Нормализуем данные.
columns_to_normalize = ["Age", "SibSp", "Parch", "Fare"]
scaler: StandardScaler = StandardScaler()
scaler.fit(train_df[columns_to_normalize])

train_norm_df = train_df
test_norm_df = test_df

train_norm_df[columns_to_normalize] = scaler.transform(train_norm_df[columns_to_normalize])
test_norm_df[columns_to_normalize] = scaler.transform(test_norm_df[columns_to_normalize])

# Сохраняет нормализованные данные.
save_results(current_script_name, train_norm_df, "train_norm_df")
save_results(current_script_name, test_norm_df, "test_norm_df")
