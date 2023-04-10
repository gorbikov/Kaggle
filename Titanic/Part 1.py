import pathlib

import pandas as pd

from eda import *

# Вводные для удобства.
current_script_name = pathlib.Path(__file__).name
random_ceed = 777

# Создаёт датафрейм.
train_path = pathlib.Path("data/train.csv")
original_train_df = pd.read_csv(train_path, index_col="PassengerId")
test_path = pathlib.Path("data/test.csv")
original_test_df = pd.read_csv(test_path, index_col="PassengerId")

# Просмотр первоначальных данных.
inspect_data(current_script_name, original_train_df, "original_train_df")
inspect_data(current_script_name, original_test_df, "original_test_df")

count_unique(original_train_df, "original_test_df")
count_unique(original_test_df, "original_test_df")

for column in original_train_df[["Age", "Fare"]].columns:
    generate_boxplot(current_script_name, original_train_df, "original_train_df", column)
    generate_boxplot(current_script_name, original_test_df, "original_test_df", column)

for column in original_train_df[["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Pclass"]].columns:
    generate_histogram(current_script_name, original_train_df, "original_train_df", column)
    generate_histogram(current_script_name, original_test_df, "original_test_df", column)

generate_histogram(current_script_name, original_train_df, "original_train_df", "Survived")

# Обработка первоначальных данных.
show_separator("Обработка данных.", size="large")
train_df: pd.DataFrame = original_train_df.drop(labels=["Name", "Ticket", "Cabin"], axis=1)
test_df: pd.DataFrame = original_test_df.drop(labels=["Name", "Ticket", "Cabin"], axis=1)
show_separator("Столбцы Name, Ticket, Cabin удалены.")

age_df = pd.concat([train_df["Age"], test_df["Age"]], axis=0).dropna(axis=0)
mean_for_age = round(age_df.mean(), 0)
train_df["Age"] = train_df["Age"].fillna(mean_for_age)
test_df["Age"] = test_df["Age"].fillna(mean_for_age)
show_separator("NA в столбце Age заменены на средний возраст: " + str(mean_for_age))

embarked_df = pd.concat([train_df["Embarked"], test_df["Embarked"]], axis=0).dropna(axis=0)
mode_for_embarked = embarked_df.mode()[0]
train_df["Embarked"] = train_df["Embarked"].fillna(mode_for_embarked)
test_df["Embarked"] = test_df["Embarked"].fillna(mode_for_embarked)
show_separator("NA в столбце Embarked заменены на моду: " + str(mode_for_embarked))

fare_df = pd.concat([train_df["Fare"], test_df["Fare"]], axis=0).dropna(axis=0)
mean_for_fare = round(fare_df.median(), 4)
train_df["Fare"] = train_df["Fare"].fillna(mean_for_fare)
test_df["Fare"] = test_df["Fare"].fillna(mean_for_fare)
show_separator("NA в столбце Fare заменены на медиану: " + str(mean_for_fare))

for column in train_df[["Pclass", "Embarked"]]:
    train_df = pd.concat([train_df.drop(column, axis=1), pd.get_dummies(train_df[column], prefix=column)], axis=1)
    test_df = pd.concat([test_df.drop(column, axis=1), pd.get_dummies(test_df[column], prefix=column)], axis=1)
for column in train_df[["Sex"]]:
    train_df = pd.concat(
        [train_df.drop(column, axis=1), pd.get_dummies(train_df[column], prefix=column, drop_first=True)], axis=1)
    test_df = pd.concat([test_df.drop(column, axis=1), pd.get_dummies(test_df[column], prefix=column, drop_first=True)],
                        axis=1)
show_separator("Столбцы Pclass, Embarked и Sex заменены на dummies.")

search_duplicates(original_train_df, "original_train_df")
search_duplicates(original_test_df, "original_test_df")

show_nans(train_df, "train_df")
show_nans(test_df, "test_df")

# Просмотр линейных зависимостей.
generate_correlation_with_target(current_script_name, train_df, "train_df", "Survived")

save_results(current_script_name, train_df, "train_df")
save_results(current_script_name, test_df, "test_df")