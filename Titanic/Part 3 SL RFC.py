import pathlib

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from eda import *

# Вводные для удобства.
current_script_name = pathlib.Path(__file__).name
random_ceed = 777
results_folder_path = pathlib.Path("intermediate data/results/")

# Загружает данные и создаёт датафреймы.
train_norm_df_path = pathlib.Path("intermediate data/results/Part 2.py_train_norm_df.csv")
test_norm_df_path = pathlib.Path("intermediate data/results/Part 2.py_test_norm_df.csv")

train_norm_df = pd.read_csv(train_norm_df_path, index_col="PassengerId")
test_norm_df = pd.read_csv(test_norm_df_path, index_col="PassengerId")

# Собирает модель метода базисных векторов.
clf: RandomForestClassifier = RandomForestClassifier()
show_separator('Параметры классификатора.')
print(clf.get_params())

# Подбирает лучшие параметры модели.
param_grid = {'n_estimators': [275],
              'max_depth': [10],
              'max_features': [11],
              'criterion': ['log_loss']}
clf_gscv = GridSearchCV(clf, param_grid, cv=4, verbose=4)
clf_gscv.fit(train_norm_df.drop(['Survived'], axis=1),
             train_norm_df[['Survived']].values.ravel())

show_separator("Лучшие параметры модели.")
print(clf_gscv.best_params_)

# Загружает лучшие параметры и тренирует модель.
clf: RandomForestClassifier = RandomForestClassifier(n_estimators=clf_gscv.best_params_['n_estimators'],
                                                     max_depth=clf_gscv.best_params_['max_depth'],
                                                     max_features=clf_gscv.best_params_['max_features'],
                                                     criterion=clf_gscv.best_params_['criterion'])
clf.fit(train_norm_df.drop(['Survived'], axis=1), train_norm_df[['Survived']].values.ravel())

# Оценка на тренировочных данных.
show_separator("Оценка на тренировочных данных.")
print(clf.score(train_norm_df.drop(['Survived'], axis=1), train_norm_df[['Survived']].values.ravel()))

# Делаем прогноз.
target_predicted_df = pd.DataFrame(clf.predict(test_norm_df), columns=['Survived'])

# Приводим прогноз в соответствие с условиями задачи.
target_predicted_df["PassengerId"] = test_norm_df.index
target_predicted_df = target_predicted_df.set_index('PassengerId')
target_predicted_df["Survived"] = target_predicted_df["Survived"].astype(int)
target_predicted_df.info()

save_results(current_script_name, target_predicted_df, "target_predicted_df")

generate_histogram(current_script_name, target_predicted_df, "target_predicted_df", "Survived")
