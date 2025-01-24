
# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm
import seaborn as sns  # Для улучшенных визуализаций
import pandas as pd  # Для структурированного анализа данных
import os  # Для работы с файловой системой
from torchvision.utils import make_grid  # Для вывода сетки изображений

# Дополнительные настройки визуализации
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Для фиксирования случайного поведения
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    np.random.seed(seed)
    print(f"Seed установлен: {seed}")

# Рассчёт значения функции потерь
def eval_loss(loader, device, net, criterion):
    """
    Рассчитывает значение функции потерь для одного батча данных.
    """
    # Получаем данные из загрузчика
    for images, labels in loader:
        break

    # Переносим данные на устройство
    inputs = images.to(device)
    labels = labels.to(device)

    # Рассчитываем предсказания
    outputs = net(inputs)

    # Вычисляем потери
    loss = criterion(outputs, labels)

    return loss

# Функция для обучения модели
def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    """
    Основная функция обучения. Поддерживает тренировку и валидацию на каждой эпохе.
    """
    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs + base_epochs):
        # Переменные для накопления результатов
        n_train_acc, n_val_acc = 0, 0
        train_loss, val_loss = 0, 0
        n_train, n_test = 0, 0

        # --- Тренировочная фаза ---
        net.train()
        for inputs, labels in tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{num_epochs + base_epochs}"):
            train_batch_size = len(labels)
            n_train += train_batch_size

            # Перенос данных на устройство
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Сбрасываем градиенты
            optimizer.zero_grad()

            # Вычисляем предсказания
            outputs = net(inputs)

            # Считаем потери
            loss = criterion(outputs, labels)
            loss.backward()

            # Обновляем параметры
            optimizer.step()

            # Подсчёт точных предсказаний
            predicted = torch.max(outputs, 1)[1]
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()

        # --- Фаза валидации ---
        net.eval()
        with torch.no_grad():
            for inputs_test, labels_test in test_loader:
                test_batch_size = len(labels_test)
                n_test += test_batch_size

                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = net(inputs_test)
                loss_test = criterion(outputs_test, labels_test)

                predicted_test = torch.max(outputs_test, 1)[1]
                val_loss += loss_test.item() * test_batch_size
                n_val_acc += (predicted_test == labels_test).sum().item()

        # Рассчёт средней потери и точности
        train_acc = n_train_acc / n_train
        val_acc = n_val_acc / n_test
        avg_train_loss = train_loss / n_train
        avg_val_loss = val_loss / n_test

        # Логирование
        print(f"Эпоха [{epoch + 1}/{num_epochs + base_epochs}], Потери: {avg_train_loss:.5f}, Точность: {train_acc:.5f}, "
              f"Валидация: Потери: {avg_val_loss:.5f}, Точность: {val_acc:.5f}")
        
        # Обновление истории
        history = np.vstack((history, np.array([epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc])))

    return history

# Визуализация истории обучения
def evaluate_history(history, model_name="Model"):
    """
    Отображает графики потерь и точности, с заголовком для идентификации модели.
    """
    print(f"[{model_name}]")
    print(f"Начальные значения: Потери: {history[0, 3]:.5f}, Точность: {history[0, 4]:.5f}")
    print(f"Конечные значения: Потери: {history[-1, 3]:.5f}, Точность: {history[-1, 4]:.5f}")

    num_epochs = len(history)
    unit = max(1, num_epochs // 10)

    # График потерь
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 0], history[:, 1], label="Тренировочные потери")
    plt.plot(history[:, 0], history[:, 3], label="Валидационные потери")
    plt.title(f"Потери ({model_name})")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    plt.grid(True)
    plt.show()

    # График точности
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 0], history[:, 2], label="Тренировочная точность")
    plt.plot(history[:, 0], history[:, 4], label="Валидационная точность")
    plt.title(f"Точность ({model_name})")
    plt.xlabel("Эпохи")
    plt.ylabel("Точность")
    plt.legend()
    plt.grid(True)
    plt.show()

# Визуализация изображений и предсказаний
def show_images_labels(loader, classes, net, device, model_name="Model"):
    """
    Отображает изображения и их предсказания, с указанием имени модели.
    """
    for images, labels in loader:
        break

    n_size = min(len(images), 50)

    if net is not None:
        inputs = images.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        predicted = torch.max(outputs, 1)[1]

    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Результаты предсказаний ({model_name})", fontsize=16)
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)
        label_name = classes[labels[i]]
        if net is not None:
            predicted_name = classes[predicted[i]]
            color = 'green' if label_name == predicted_name else 'red'
            ax.set_title(f"{label_name}:{predicted_name}", color=color, fontsize=10)
        else:
            ax.set_title(label_name, fontsize=10)

        img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
        img = (img + 1) / 2
        plt.imshow(img)
        ax.axis('off')
    plt.show()
