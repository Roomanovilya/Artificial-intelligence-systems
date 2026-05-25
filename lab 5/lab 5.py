import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

moscow_flats = pd.read_csv('ml_moscow_flats.csv')

x = moscow_flats[['wallsMaterial', 'floorNumber', 'floorsTotal', 'totalArea', 'kitchenArea', 'latitude', 'longitude']]
y = moscow_flats['price'].values
xe = pd.get_dummies(x, columns=['wallsMaterial'], drop_first=True, dtype=int)
x = xe.values

# Предобработка
# считаем среднее и отклонение по таблице
def calculate_scaling_params(X):
    mean = np.mean(X, axis=0)  # Среднее арифметическое для каждого столбца
    std = np.std(X, axis=0)    # Стандартное отклонение для каждого столбца
    # Защита от деления на ноль 
    std[std == 0] = 1.0
    return mean, std

# мастштабируем данные по формуле
def apply_scaling(X, mean, std):
    # Математическая формула Z-score: (значение - среднее) / отклонение
    return (X - mean) / std

# избавляемся от выбросов 
total_area_raw = moscow_flats['totalArea'].values

q1_area, q3_area = np.percentile(total_area_raw, [25, 75])
iqr_area = q3_area - q1_area
lower_area = q1_area - 1.5 * iqr_area
upper_area = q3_area + 1.5 * iqr_area

q1_price, q3_price = np.percentile(y, [25, 75])
iqr_price = q3_price - q1_price
lower_price = q1_price - 1.5 * iqr_price
upper_price = q3_price + 1.5 * iqr_price

mask = (total_area_raw >= lower_area) & (total_area_raw <= upper_area) & \
       (y >= lower_price) & (y <= upper_price)

x_clean = x[mask]
mean, std = calculate_scaling_params(x_clean)
x_norm = apply_scaling(x_clean, mean, std)
y_norm = y[mask]

poly = PolynomialFeatures(degree=2, include_bias=False)
x_norm = poly.fit_transform(x_norm)

# Настройка кросс-валидации (5 фолдов)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# разделение на обучающую и тестовую выборку (80/20)
# Разделяем сырые данные
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Разделяем предобработанные данные
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
    x_norm, y_norm, test_size=0.2, random_state=42
)

print("Результаты на сырых данных")

# Orthogonal Matching Pursuit 
t0 = time.perf_counter()
omp_raw = OrthogonalMatchingPursuit()
omp_raw.fit(X_train_raw, y_train_raw)
y_pred_omp_raw = omp_raw.predict(X_test_raw)
time_omp_raw = time.perf_counter() - t0

# MLPRegressor
t0 = time.perf_counter()
mlp_raw = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)
mlp_raw.fit(X_train_raw, y_train_raw)
y_pred_mlp_raw = mlp_raw.predict(X_test_raw)
time_mlp_raw = time.perf_counter() - t0

accuracy_omp_raw = r2_score(y_test_raw, y_pred_omp_raw)
mae_omp_raw = mean_absolute_error(y_test_raw, y_pred_omp_raw)
accuracy_mlp_raw = r2_score(y_test_raw, y_pred_mlp_raw)
mae_mlp_raw = mean_absolute_error(y_test_raw, y_pred_mlp_raw)

print(f"OMP Время: {time_omp_raw:.4f} сек | Точность : {accuracy_omp_raw:.4f} | MAE: {mae_omp_raw:,.0f} руб.")
print(f"MLP Время: {time_mlp_raw:.4f} сек | Точность : {accuracy_mlp_raw:.4f} | MAE: {mae_mlp_raw:,.0f} руб.")

print("\nРезультаты на предобработанных данных")

# Кросс-валидация на предобработанных данных
cv_omp_norm = cross_val_score(OrthogonalMatchingPursuit(), x_norm, y_norm, cv=cv, scoring='r2')
cv_mlp_norm = cross_val_score(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42), x_norm, y_norm, cv=cv, scoring='r2')

print(f"OMP Точность фолдам: {[f'{val:.4f}' for val in cv_omp_norm]} | Средняя точность CV: {cv_omp_norm.mean():.4f}")
print(f"MLP Точность фолдам: {[f'{val:.4f}' for val in cv_mlp_norm]} | Средняя точность CV: {cv_mlp_norm.mean():.4f}\n")

# Orthogonal Matching Pursuit 
t0 = time.perf_counter()
omp_norm = OrthogonalMatchingPursuit()
omp_norm.fit(X_train_norm, y_train_norm)
y_pred_omp_norm = omp_norm.predict(X_test_norm)
time_omp_norm = time.perf_counter() - t0

# MLPRegressor
t0 = time.perf_counter()
mlp_norm = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)
mlp_norm.fit(X_train_norm, y_train_norm)
y_pred_mlp_norm = mlp_norm.predict(X_test_norm)
time_mlp_norm = time.perf_counter() - t0

accuracy_omp_norm = r2_score(y_test_norm, y_pred_omp_norm)
mae_omp_norm = mean_absolute_error(y_test_norm, y_pred_omp_norm)
accuracy_mlp_norm = r2_score(y_test_norm, y_pred_mlp_norm)
mae_mlp_norm = mean_absolute_error(y_test_norm, y_pred_mlp_norm)

print(f"OMP Время: {time_omp_norm:.4f} сек | Точность : {accuracy_omp_norm:.4f} | MAE: {mae_omp_norm:,.0f} руб.")
print(f"MLP Время: {time_mlp_norm:.4f} сек | Точность : {accuracy_mlp_norm:.4f} | MAE: {mae_mlp_norm:,.0f} руб.")


# Графики 
plt.rcParams['font.size'] = 11

# OMP на СЫРЫХ данных 
plt.figure(figsize=(7, 5))
max_val = max(y_test_raw.max(), y_pred_omp_raw.max())
plt.scatter(y_test_raw, y_pred_omp_raw, alpha=0.3, s=15, c='steelblue')
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеал')
plt.xlabel('Реальная цена ')
plt.ylabel('Предсказанная цена ')
plt.title(f'Orthogonal Matching Pursuit\nТочность: {accuracy_omp_raw:.4f} | MAE: {mae_omp_raw:,.0f} руб')
plt.ticklabel_format(style='scientific', axis='both', scilimits=(6,6))
plt.legend()
plt.tight_layout()
plt.show()

# MLP на СЫРЫХ данных 
plt.figure(figsize=(7, 5))
max_val = max(y_test_raw.max(), y_pred_mlp_raw.max())
plt.scatter(y_test_raw, y_pred_mlp_raw, alpha=0.3, s=15, c='darkgreen')
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеал')
plt.xlabel('Реальная цена ')
plt.ylabel('Предсказанная цена ')
plt.title(f'MLPRegressor\nТочность: {accuracy_mlp_raw:.4f} | MAE: {mae_mlp_raw:,.0f} руб')
plt.ticklabel_format(style='scientific', axis='both', scilimits=(6,6))
plt.legend()
plt.tight_layout()
plt.show()

# OMP на ПРЕДОБРАБОТАННЫХ данных 
plt.figure(figsize=(7, 5))
max_val = max(y_test_norm.max(), y_pred_omp_norm.max())
plt.scatter(y_test_norm, y_pred_omp_norm, alpha=0.3, s=15, c='steelblue')
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеал')
plt.xlabel('Реальная цена ')
plt.ylabel('Предсказанная цена ')
plt.title(f'Orthogonal Matching Pursuit\nТочность: {accuracy_omp_norm:.4f} | MAE: {mae_omp_norm:,.0f} руб')
plt.ticklabel_format(style='scientific', axis='both', scilimits=(6,6))
plt.legend()
plt.tight_layout()
plt.show()

# MLP на ПРЕДОБРАБОТАННЫХ данных
plt.figure(figsize=(7, 5))
max_val = max(y_test_norm.max(), y_pred_mlp_norm.max())
plt.scatter(y_test_norm, y_pred_mlp_norm, alpha=0.3, s=15, c='darkgreen')
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеал')
plt.xlabel('Реальная цена ')
plt.ylabel('Предсказанная цена ')
plt.title(f'MLPRegressor\nТочность: {accuracy_mlp_norm:.4f} | MAE: {mae_mlp_norm:,.0f} руб')
plt.ticklabel_format(style='scientific', axis='both', scilimits=(6,6))
plt.legend()
plt.tight_layout()
plt.show()