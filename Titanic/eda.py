from pathlib import Path

import matplotlib.axes
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_separator(*title_texts: str, size='small'):
    """Выводит разделитель в виде штриховой линии."""
    match size:
        case "small":
            print()
            for title_text in title_texts:
                print(title_text)
            print("---------------------------------------------------------------------")
        case "large":
            print()
            print()
            for title_text in title_texts:
                print(title_text)
            print("===================================================================================================")


def save_results(current_script_name: str, df: pd.DataFrame, filename: str):
    """Сохраняет датафрейм в виде csv в папку intermediate data/results/."""
    show_separator("Сохраняем датафрейм " + filename + " в csv")
    filepath = Path(str("intermediate data/results/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(filename + " сохранен в " + str(filepath))


def save_to_tmp(current_script_name: str, df: pd.DataFrame, filename: str):
    """Сохраняет датафрейм в виде csv в папку intermediate data/tmp/."""
    show_separator("Сохраняем датафрейм " + filename + " в csv")
    filepath = Path(str("intermediate data/tmp/" + current_script_name + "_" + filename + '.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)
    print(filename + " сохранен в " + str(filepath))


def inspect_data(current_script_name: str, df: pd.DataFrame, filename: str):
    """Выводит инфо, сохраняет голову датафрейма в csv в папку intermediate data/heads/."""
    show_separator("Информация по " + filename)
    # Выводит инфо.
    df.info()
    # Сохраняет голову в файл.
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_head.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.head().to_csv(filepath)
    # Сохраняет статистику в файл.
    filepath = Path(str("intermediate data/heads/" + current_script_name + "_" + filename + '_describe.csv'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.describe().to_csv(filepath)


def search_duplicates(df: pd.DataFrame, df_name: str):
    """Выводит количество дубликатов в датафрейме."""
    show_separator("Поиск дубликатов в " + df_name)
    print("Количество дубликатов в " + df_name + ":")
    print(df[df.duplicated()].shape[0])


def delete_duplicates(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    """Удаляет дубликаты в датафрейме."""
    show_separator("Удаление дубликатов в " + df_name)
    print("Размер " + df_name + " до удаления дубликатов: " + str(df.shape))
    df = df.drop_duplicates()
    print("Размер " + df_name + " после удаления дубликатов: " + str(df.shape))
    return df


def show_nans(df: pd.DataFrame, df_name: str):
    """Выводит количество пустых клеток в датафрейме."""
    show_separator("Поиск пустых клеток в " + df_name)
    print("Количество строк с пустыми клетками в " + df_name + ":")
    print(df[df.isnull().any(axis=1)].shape[0])
    print("Количество столбцов с пустыми клетками в " + df_name + ":")
    print(df.loc[:, df.isnull().any()].columns.size)


def count_unique_for_object_type(df: pd.DataFrame, df_name: str):
    """Выбирает столбцы типа object, выводит количество уникальных записей в каждом таком столбце."""
    show_separator("Уникальные значения в столбцах типа object в " + df_name)
    dummy_counter = 0
    for col in df.columns:
        if df[col].dtypes == object:
            dummy_counter += len(df[col].unique())
            print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))

    print('Dummy columns: ' + str(dummy_counter))


def count_unique(df: pd.DataFrame, df_name: str):
    """Выводит количество уникальных записей в каждом столбце."""
    show_separator("Уникальные значения в столбцах в " + df_name)
    for col in df.columns:
        print('Unique in ' + str(col) + ': ' + str(len(df[col].unique())))


def generate_boxplot(current_script_name: str, df: pd.DataFrame, df_name: str, column_name: str):
    """Сохраняет график с выбросами в папку "intermediate data/diagrams/."""

    # Выводит разделитель с описанием того, что делает функция.
    show_separator("Боксплот для столбца " + column_name + " в датафрейме " + df_name)

    # Выбирает столбец для анализа.
    current_column = df[[column_name]].dropna()

    # Создаёт фигуру, оси и решетку.
    fig: matplotlib.figure.Figure = plt.figure()
    ax: matplotlib.axes.Axes = plt.axes()
    fig.set_size_inches(w=19.2, h=10.8)
    ax.grid()

    # Рисует график.
    ax.set_title("Распределение значений для столбца " + column_name + " в датафрейме " + df_name)
    ax.boxplot(current_column, showmeans=True)

    # Добавляет статистику на график.
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.text(x_min + (x_max - x_min) / 100, y_min + (y_max - y_min) / 100, str(df[[column_name]].describe()),
            bbox={'facecolor': 'grey', 'edgecolor': 'None', 'alpha': 0.5, 'pad': 0.3})

    # Регулирует отступы на графике.
    plt.tight_layout()

    # Сохраняет график в файл.
    filepath = Path(
        str("intermediate data/diagrams/" + current_script_name + "_" + df_name + "_" + column_name + '_boxplot.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print("Сохранено в intermediate data/diagrams/")

    # Убирает фигуру из памяти и закрывает график.
    fig.clear()
    plt.close()


def generate_histogram(current_script_name: str, df: pd.DataFrame, df_name: str, column_name: str, bins: int = 50):
    """Сохраняет гистограмму в папку "intermediate data/diagrams/."""

    # Выводит разделитель с описанием того, что делает функция.
    show_separator("Гистограмма для столбца " + column_name + " в датафрейме " + df_name)

    # Выбирает столбец для анализа.
    current_column = df[[column_name]].dropna()

    # Создаёт фигуру, оси и решетку.
    fig: matplotlib.figure.Figure = plt.figure()
    ax: matplotlib.axes.Axes = plt.axes()
    fig.set_size_inches(w=19.2, h=10.8)
    ax.grid()

    # Рисует график.
    ax.set_title("Гистограмма для столбца " + column_name + " в датафрейме " + df_name)
    ax.hist(x=current_column, bins=bins)

    # Добавляет подписи данных.
    graph_patches = ax.patches
    x_pos = np.arange(len(graph_patches))
    for patch, value in zip(graph_patches, x_pos):
        patch: matplotlib.patches.Rectangle = patch
        width = patch.get_width()
        height = patch.get_height()
        if height != 0:
            ax.text(patch.get_x() + width / 2, patch.get_height() / 2, round(height, 2), horizontalalignment='center',
                    bbox={'facecolor': 'grey', 'edgecolor': 'None', 'alpha': 0.5, 'pad': 0.3})

    # Регулирует отступы на графике.
    plt.tight_layout()

    # Сохраняет график в файл.
    filepath = Path(
        str("intermediate data/diagrams/" + current_script_name + "_" + df_name + "_" + column_name + '_histogram.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print("Сохранено в intermediate data/diagrams/")

    # Убирает фигуру из памяти и закрывает график.
    fig.clear()
    plt.close()


def generate_loss_function_graph(current_script_name: str, data: dict):
    """Сохраняет график loss function в папку intermediate data/diagrams/."""

    # Выводит разделитель с описанием того, что делает функция.
    show_separator("Сохраняем график loss function в папку intermediate data/diagrams/.")

    # Создаёт фигуру, оси и решетку.
    fig: matplotlib.figure.Figure = plt.figure()
    ax: matplotlib.axes.Axes = plt.axes()
    fig.set_size_inches(w=19.2, h=10.8)
    ax.grid()

    # Рисует график.
    ax.set_title("График loss function в процессе обучения.")
    ax.plot(data.keys(), data.values())

    # Добавляет подписи данных.
    for key in data.keys():
        ax.text(key, data[key], str(round(data[key], 5)), horizontalalignment='center',
                bbox={'facecolor': 'grey', 'edgecolor': 'None', 'alpha': 0.5, 'pad': 0.3})

    # Регулирует отступы на графике.
    plt.tight_layout()

    # Сохраняет график в файл.
    filepath = Path(
        str("intermediate data/diagrams/" + current_script_name + '_loss_function.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print("Сохранено в intermediate data/diagrams/")

    # Убирает фигуру из памяти и закрывает график.
    fig.clear()
    plt.close()


def generate_correlation_with_target(current_script_name: str, df: pd.DataFrame, df_name: str,
                                     column_for_correlation: str):
    """Выводит матрицу корреляций для датафрейма + сохраняет столбчатый график в папку intermediate data/diagrams/."""

    # Выводит разделитель с описанием того, что делает функция.
    show_separator("Матрица корреляций для " + df_name + " со столбцом " + column_for_correlation)

    # Считает и выводит матрицу корреляций.
    corr_df = df.corr()[[column_for_correlation]].drop(column_for_correlation, axis=0)
    print(corr_df.sort_values(by=[column_for_correlation], ascending=False))

    # Формирует данные для составления графика.
    plt_labels = corr_df.index.values
    plt_values = corr_df[[column_for_correlation]].values.reshape(plt_labels.shape[0])

    # Создаёт фигуру, оси и решетку.
    fig: matplotlib.figure.Figure = plt.figure()
    ax: matplotlib.axes.Axes = plt.axes()
    fig.set_size_inches(w=19.2, h=10.8)
    ax.grid()

    # Рисует график.
    ax.set_title("Матрица корреляций для " + df_name + " со столбцом " + column_for_correlation)
    y_pos = np.arange(len(plt_labels))
    ax.barh(y_pos, plt_values)
    ax.set_yticks(y_pos, plt_labels)

    # Добавляет подписи данных.
    graph_patches = ax.patches
    for patch, value in zip(graph_patches, plt_values):
        patch: matplotlib.patches.Rectangle = patch
        width = patch.get_width()
        ax.text(width / 2, patch.get_y() + patch.get_height() / 2, round(value, 2), verticalalignment='center',
                bbox={'facecolor': 'grey', 'edgecolor': 'None', 'alpha': 0.5, 'pad': 0.3})

    # Регулирует отступы на графике.
    plt.tight_layout()

    # Сохраняет график в файл.
    filepath = Path(str("intermediate data/diagrams/" + current_script_name + "_" + df_name + '_correlation.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print("Сохранено в intermediate data/diagrams/")

    # Убирает фигуру из памяти и закрывает график.
    fig.clear()
    plt.close()


def generate_correlation_heatmap(current_script_name: str, df: pd.DataFrame, df_name: str):
    """Сохраняет тепловую матрицу корреляций для датафрейма в папку intermediate data/diagrams/."""

    # Выводит разделитель с описанием того, что делает функция.
    show_separator("Тепловая карта корреляций для " + df_name)

    # Формирует данные для составления графика.
    corr_df = df.corr()
    print(corr_df)
    plt_labels = corr_df.columns

    # Создаёт фигуру, оси и решетку.
    fig: matplotlib.figure.Figure = plt.figure()
    ax: matplotlib.axes.Axes = plt.axes()
    fig.set_size_inches(w=19.2, h=10.8)

    # Рисует график.
    ax.set_title("Тепловая карта корреляций для " + df_name)
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(plt_labels, rotation=65)
    ax.set_yticklabels(plt_labels, rotation=0)
    im = ax.imshow(corr_df)
    fig.colorbar(im, orientation='vertical', fraction=0.05)

    # Добавляет подписи данных.
    for i in range(len(corr_df.columns)):
        for j in range(len(corr_df.columns)):
            ax.text(i, j, round(corr_df.iloc[j, i], 2), ha="center", va="center", color="w")


    # Регулирует отступы на графике.
    plt.tight_layout()

    # Сохраняет график в файл.
    filepath = Path(str("intermediate data/diagrams/" + current_script_name + "_" + df_name + '_heatmap.png'))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    print("Сохранено в intermediate data/diagrams/")

    # Убирает фигуру из памяти и закрывает график.
    fig.clear()
    plt.close()
