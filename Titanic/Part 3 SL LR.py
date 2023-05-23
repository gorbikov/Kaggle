import pathlib
from os import path

import torch
import torch.nn

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

# Создаёт тензоры из датафреймов.
test_norm_tensor: torch.Tensor = torch.Tensor(test_norm_df.values)
train_norm_tensor: torch.Tensor = torch.Tensor(train_norm_df.drop(["Survived"], axis=1).values)
train_target_tensor: torch.Tensor = torch.Tensor(train_norm_df[["Survived"]].values)


# Собирает модель логистической регрессии.
# Задаёт параметры обучения.
num_epochs = 100000
learning_rate = 0.001
# Использует Binary Cross Entropy.
# Использует ADAM optimizer.
# Загружает модель из файла.
# Начинает обучение.
# Сохраняет в файл график loss function.
generate_loss_function_graph(current_script_name, loss_function_values_for_graph)

# Сохраняет параметры модели в файл.
torch.save(lr.state_dict(), results_folder_path.joinpath(current_script_name + '_model_weights'))
show_separator("Параметры модели сохранены в папке results.")

target_predicted: torch.Tensor = lr(test_norm_tensor).round()

target_predicted_df = pd.DataFrame(target_predicted.detach().numpy()).rename(columns={0: "Survived"})
target_predicted_df["PassengerId"] = test_norm_df.index
target_predicted_df = target_predicted_df.set_index('PassengerId')
target_predicted_df["Survived"] = target_predicted_df["Survived"].astype(int)
target_predicted_df.info()

save_results(current_script_name, target_predicted_df, "target_predicted_df")

generate_histogram(current_script_name, target_predicted_df, "target_predicted_df", "Survived")
