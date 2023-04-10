import pathlib
from os import path

import torch
import torch.nn
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
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
class LogisticRegression(torch.nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_input_features, 1)

    # sigmoid transformation of the input
    def forward(self, x):
        y_prediction = torch.sigmoid(self.linear(x))
        return y_prediction


lr = LogisticRegression(train_norm_tensor.size()[1])

# Задаёт параметры обучения.
num_epochs = 100000
learning_rate = 0.001
# Использует Binary Cross Entropy.
criterion = torch.nn.BCELoss()
# Использует ADAM optimizer.
optimizer = torch.optim.SGD(lr.parameters(), lr=learning_rate)

# Загружает модель из файла.
if path.exists(results_folder_path.joinpath(current_script_name + '_model_weights')):
    lr.load_state_dict(torch.load(results_folder_path.joinpath(current_script_name + '_model_weights')))
    show_separator("Параметры модели загружены из файла.")
else:
    show_separator("Файл с параметрами подели отсутствует. Обучение начинается с нуля.")

# Начинает обучение.
show_separator("Обучение модели на " + str(num_epochs) + " эпохах:")
loss_function_values_for_graph = dict()
previous_loss_function_value = None
for epoch in range(num_epochs):
    y_pred = lr(train_norm_tensor)
    loss = criterion(y_pred, train_target_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 100 == 0:
        # Выводит loss function каждый 20 эпох.
        loss_function_values_for_graph[epoch + 1] = loss.item()
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
    if (previous_loss_function_value is not None) and (float(loss.item()) > previous_loss_function_value):
        show_separator("!!!Обучение остановлено, т.к. зафиксирован рост lost function.!!!")
        break
    previous_loss_function_value = float(loss.item())

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
