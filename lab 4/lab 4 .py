import numpy as np
import pandas as pd
import time
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

# Загрузка данных

train_df = pd.read_csv('disease_train.csv')
test_df = pd.read_csv('disease_public_test.csv')
submission_df = pd.read_csv('disease_sample_submission.csv')

X_train = train_df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values
X_test = test_df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']].values

y_train = train_df['Y'].values
y_test = submission_df['Y'].values

# считаем среднее и отклонение по таблице
def calculate_scaling_params(X):
    mean = np.mean(X, axis=0)  # Среднее арифметическое для каждого столбца
    std = np.std(X, axis=0)    # Стандартное отклонение для каждого столбца
    # Защита от деления на ноль 
    std[std == 0] = 1.0
    return mean, std

# мастштабируем данные по готовой формуле
def apply_scaling(X, mean, std):
    # Математическая формула Z-score: (значение - среднее) / отклонение
    return (X - mean) / std

# Находим среднее и отклонение по обучающим данным
train_mean, train_std = calculate_scaling_params(X_train)

# Масштабируем обучающие данные 
X_train_recycled = apply_scaling(X_train, train_mean, train_std)

# Масштабируем тестовые данные 
X_test_recycled = apply_scaling(X_test, train_mean, train_std)

# Ручной DummyClassifier (Стратегия "Пропорционально")

def probability_one(y):
    # Посчитаем общую длину выборки
    total = len(y)
    
    # Посчитаем количество единиц 
    counts_1 = np.sum(y == 1)
    
    # Находим вероятность встретить 1
    prob_one = counts_1 / total 
    
    # Возвращаем эту вероятность
    return prob_one


def DummyClassifier_manual(prob_one, X_matrix):
    # Количество x, для которых нужно сделать прогноз
    n_samples = X_matrix.shape[0]

    np.random.seed(42)
    
    # Генерируем случайные числа от 0.0 до 1.0
    random_numbers = np.random.rand(n_samples)
    
    predictions = np.where(random_numbers < prob_one, 1, 0)
    
    return predictions

# Ручной Perceptron

def predict(params, w, b):

    z = np.dot(params, w) + b

    return 1 if z >= 0 else 0 


# обучение Perceptron
def train_perceptron(X, y, lr=0.005, max_epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(max_epochs):
        z = np.dot(X, w) + b
        
        predicts = np.where(z >= 0, 1, 0)
        
        loss = y - predicts

        w = w + lr * np.dot(X.T, loss)
        b = b + lr * np.sum(loss)
        
    return w, b

print(f'-----Результаты(на сырых данных)-----')

# 1) Ручной Dummy
t0 = time.perf_counter()
prob_1_raw = probability_one(y_train)
y_pred_dummy_manual_raw = DummyClassifier_manual(prob_1_raw, X_test)
time_dummy_manual_raw = time.perf_counter() - t0
acc_dummy_manual_raw = accuracy_score(y_test, y_pred_dummy_manual_raw)
print(f"1. Ручной Dummy Classifier | Точность: {acc_dummy_manual_raw:.4f} | Время: {time_dummy_manual_raw:.6f} сек.")

# 2) Sklearn Dummy
t0 = time.perf_counter()
sklearn_dummy_raw = DummyClassifier(strategy='stratified', random_state=42)
sklearn_dummy_raw.fit(X_train, y_train)
y_pred_dummy_sklearn_raw = sklearn_dummy_raw.predict(X_test)
time_dummy_sklearn_raw = time.perf_counter() - t0
acc_dummy_sklearn_raw = accuracy_score(y_test, y_pred_dummy_sklearn_raw)
print(f"2. Sklearn Dummy Classifier| Точность: {acc_dummy_sklearn_raw:.4f} | Время: {time_dummy_sklearn_raw:.6f} сек.")

# 3) Ручной Perceptron 
t0 = time.perf_counter()
w_manual_raw, b_manual_raw = train_perceptron(X_train, y_train, 0.005, 1000)
y_pred_perceptron_manual_raw = [predict(x, w_manual_raw, b_manual_raw) for x in X_test]
time_perceptron_manual_raw = time.perf_counter() - t0
acc_perceptron_manual_raw = accuracy_score(y_test, y_pred_perceptron_manual_raw)
print(f"3. Ручной Perceptron       | Точность: {acc_perceptron_manual_raw:.4f} | Время: {time_perceptron_manual_raw:.6f} сек.")

# 4) Sklearn Perceptron
t0 = time.perf_counter()
sklearn_perceptron_raw = Perceptron(max_iter=1000, eta0=0.05, random_state=42)
sklearn_perceptron_raw.fit(X_train, y_train)
y_pred_perceptron_sklearn_raw = sklearn_perceptron_raw.predict(X_test)
time_perceptron_sklearn_raw = time.perf_counter() - t0
acc_perceptron_sklearn_raw = accuracy_score(y_test, y_pred_perceptron_sklearn_raw)
print(f"4. Sklearn Perceptron      | Точность: {acc_perceptron_sklearn_raw:.4f} | Время: {time_perceptron_sklearn_raw:.6f} сек.")

# 5) Sklearn KMeans
t0 = time.perf_counter()
kmeans_raw = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_raw = kmeans_raw.fit_predict(X_train)
time_kmeans_raw = time.perf_counter() - t0
sil_score_raw = silhouette_score(X_train, kmeans_raw.labels_)
print(f"5. Sklearn KMeans          | Точность: {sil_score_raw:.4f} | Время: {time_kmeans_raw:.6f} сек.")

print('')
print(f'-----Результаты(на переработанных данных)-----')

# 1) Ручной Dummy 
t0 = time.perf_counter()
prob_1_recycled = probability_one(y_train)
y_pred_dummy_manual_recycled = DummyClassifier_manual(prob_1_recycled, X_test_recycled)
time_dummy_manual_recycled = time.perf_counter() - t0
acc_dummy_manual_recycled = accuracy_score(y_test, y_pred_dummy_manual_recycled)
print(f"1. Ручной Dummy Classifier | Точность: {acc_dummy_manual_recycled:.4f} | Время: {time_dummy_manual_recycled:.6f} сек.")

# 2) Sklearn Dummy
t0 = time.perf_counter()
sklearn_dummy_recycled = DummyClassifier(strategy='stratified', random_state=42)
sklearn_dummy_recycled.fit(X_train_recycled, y_train)
y_pred_dummy_sklearn_recycled = sklearn_dummy_recycled.predict(X_test_recycled)
time_dummy_sklearn_recycled = time.perf_counter() - t0
acc_dummy_sklearn_recycled = accuracy_score(y_test, y_pred_dummy_sklearn_recycled)
print(f"2. Sklearn Dummy Classifier| Точность: {acc_dummy_sklearn_recycled:.4f} | Время: {time_dummy_sklearn_recycled:.6f} сек.")

# 3) Ручной Perceptron 
t0 = time.perf_counter()
w_manual_recycled, b_manual_recycled = train_perceptron(X_train_recycled, y_train, lr=0.05, max_epochs=1000)
y_pred_perceptron_manual_recycled = [predict(x, w_manual_recycled, b_manual_recycled) for x in X_test_recycled]
time_perceptron_manual_recycled = time.perf_counter() - t0
acc_perceptron_manual_recycled = accuracy_score(y_test, y_pred_perceptron_manual_recycled)
print(f"3. Ручной Perceptron       | Точность: {acc_perceptron_manual_recycled:.4f} | Время: {time_perceptron_manual_recycled:.6f} сек.")

# 4) Sklearn Perceptron
t0 = time.perf_counter()
sklearn_perceptron_recycled = Perceptron(max_iter=1000, eta0=0.05, random_state=42)
sklearn_perceptron_recycled.fit(X_train_recycled, y_train)
y_pred_perceptron_sklearn_recycled = sklearn_perceptron_recycled.predict(X_test_recycled)
time_perceptron_sklearn_recycled = time.perf_counter() - t0
acc_perceptron_sklearn_recycled = accuracy_score(y_test, y_pred_perceptron_sklearn_recycled)
print(f"4. Sklearn Perceptron      | Точность: {acc_perceptron_sklearn_recycled:.4f} | Время: {time_perceptron_sklearn_recycled:.6f} сек.")

# 5) Sklearn KMeans 
t0 = time.perf_counter()
kmeans_recycled = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters_recycled = kmeans_recycled.fit_predict(X_train_recycled)
time_kmeans_recycled = time.perf_counter() - t0
sil_score_recycled = silhouette_score(X_train_recycled, kmeans_recycled.labels_)
print(f"5. Sklearn KMeans          | Точность: {sil_score_recycled:.4f} | Время: {time_kmeans_recycled:.6f} сек.")

# кросс-валидация 

perceptron_experiment = Perceptron(max_iter=1000, eta0=0.05, random_state=42)

# Фиксированное разбиение 80/20 
X_tr_80, X_te_20, y_tr_80, y_te_20 = train_test_split(
    X_train_recycled, y_train, test_size=0.20, random_state=42, stratify=y_train
)
perceptron_experiment.fit(X_tr_80, y_tr_80)
preds_80_20 = perceptron_experiment.predict(X_te_20)
acc_80_20 = accuracy_score(y_te_20, preds_80_20)
print(f"Метод 80/20      | Точность: {acc_80_20:.4f}")

# кросс-валидация
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(perceptron_experiment, X_train_recycled, y_train, cv=cv_stratified, scoring='accuracy')
acc_cv_mean = cv_scores.mean()
print(f"Кросс-валидация  | Средняя точность: {acc_cv_mean:.4f}")

# Граф результаты 

# 1. Собираем все названия и метки для оси X
models = ['Dummy\n(Manual)', 'Dummy\n(Sklearn)', 'Perceptron\n(Manual)', 'Perceptron\n(Sklearn)', 'KMeans\n(Sklearn)']
x = np.arange(len(models)) 
width = 0.35 

# 2. Собираем точность из твоих переменных в списки
accuracy_raw = [acc_dummy_manual_raw, acc_dummy_sklearn_raw, acc_perceptron_manual_raw, acc_perceptron_sklearn_raw, sil_score_raw]
accuracy_recycled = [acc_dummy_manual_recycled, acc_dummy_sklearn_recycled, acc_perceptron_manual_recycled, acc_perceptron_sklearn_recycled, sil_score_recycled]

# 3. Собираем время работы из твоих переменных в списки
time_raw = [time_dummy_manual_raw, time_dummy_sklearn_raw, time_perceptron_manual_raw, time_perceptron_sklearn_raw, time_kmeans_raw]
time_recycled = [time_dummy_manual_recycled, time_dummy_sklearn_recycled, time_perceptron_manual_recycled, time_perceptron_sklearn_recycled, time_kmeans_recycled]

# --- ГРАФИК 1: ТОЧНОСТЬ МОДЕЛЕЙ ---
plt.figure(figsize=(9, 5))
plt.bar(x - width/2, accuracy_raw, width, label='Сырые данные', color='#e74c3c')
plt.bar(x + width/2, accuracy_recycled, width, label='Переработанные данные', color='#2ecc71')

plt.ylabel('Точность ')
plt.title('Сравнение точности моделей ')
plt.xticks(x, models)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()


# --- ГРАФИК 2: ВРЕМЯ РАБОТЫ ---
plt.figure(figsize=(9, 5))
plt.bar(x - width/2, time_raw, width, label='Сырые данные', color='#34495e')
plt.bar(x + width/2, time_recycled, width, label='Переработанные данные', color='#3498db')

plt.yscale('log')
plt.ylabel('Время работы (секунды)')
plt.title('Сравнение времени работы моделей')
plt.xticks(x, models)
plt.grid(axis='y', linestyle='--', alpha=0.5, which="major")
plt.legend()

plt.tight_layout()
plt.show()