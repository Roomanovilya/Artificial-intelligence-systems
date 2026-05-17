from math import * 
import matplotlib.pyplot as plt
import random 
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

f = open("Datas.txt", "r", encoding="utf8")

file = []
lenclass = []

for i in f:
    file.append(i.strip().split())
for i in range(len(file)):
    if file[i][3] not in lenclass:
            lenclass.append(file[i][3])
lenclass.sort()

k = int(input("Введите кол-во k (для сравнения): "))

# Разделение выборки на обучающую (base) и тестовую (test) в соотношении 80/20
def separation(file):
    while True:  
        random.shuffle(file)
        ky = int(len(file) * 0.8)
        base = file[:ky]
        test = file[ky:]

        # Проверяем, все ли классы попали в обучающую выборку
        sbase = []
        for i in base:
            sbase.append(i[3])
        if len(set(sbase)) == len(lenclass):
            break

    data_set = {}
    for i in range(len(base)):

        if base[i][3] not in data_set.keys():
            data_set[base[i][3]] = []

        if base[i][3] in data_set.keys():
            data_set[base[i][3]].append([int(base[i][1]), int(base[i][2])])
    return base, test, data_set

# визуализация точек
def visualize_points(base, test):
    plt.figure(figsize=(8, 8))
    plt.grid(True, zorder=1)
    
    all_data = base + test

    colors = ['orange', 'blue', 'green', 'purple', 'red', 'brown', 'pink']
    
    for index, current_class in enumerate(lenclass):
        x_coords = [int(i[1]) for i in all_data if i[3] == current_class]
        y_coords = [int(i[2]) for i in all_data if i[3] == current_class]
        
        # Если точки такого класса нашлись, рисуем их
        if x_coords:
            plt.scatter(
                x_coords, 
                y_coords, 
                color=colors[index],      
                marker='o', 
                s=90,  
                label=current_class,      
                zorder=2
            )
    
    plt.title("визуализация точек")
    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")
    plt.legend(title="Классы")
    plt.show()  

matrix = [[0 for _ in lenclass] for _ in lenclass]
accu = 0

# Рукописная реализация алгоритма K-ближайших соседей
def definition(prod, data_set, k, cross_valid=False):
    global accu, matrix
    distarray = [ ] 

    # Вычисляем расстояние до каждой точки в обучающем датасете
    for i in data_set:
        for j in range(len(data_set[i])):
            dist = sqrt((int(prod[1]) - data_set[i][j][0]) ** 2 + (int(prod[2]) - data_set[i][j][1]) ** 2)
            distarray.append([dist, i])

    # Сортируем по возрастанию расстояния и берем k первых (самых близких)
    distarray.sort()
    distarray = distarray[:k]

    definitionclass = [i[1] for i in distarray]
    # Находим самый часто встречающийся класс 
    definitionclass = max(set(definitionclass), key=definitionclass.count)

    if cross_valid == False:

        if definitionclass == prod[3]:
            accu += 1

        xa = lenclass.index(prod[3])             
        xi = lenclass.index(definitionclass)     
        matrix[xa][xi] += 1

        print(f"Класс в данных: {prod[3]} -- алгоритм: {definitionclass}")
    return definitionclass

# Классификация с использованием стандартной библиотеки Scikit-Learn.
def sklearn(base, test, k, cross_valid=False):

    x_base = [[int(i[1]), int(i[2])] for i in base]
    y_base = [i[3] for i in base]
    x_test = [[int(i[1]), int(i[2])] for i in test]
    y_test = [i[3] for i in test]  

    # all_x = [[int(i[1]), int(i[2])] for i in (base + test)]
    # all_y = [i[3] for i in (base + test)]
    # x_base, x_test, y_base, y_test = train_test_split(all_x, all_y, test_size=0.2)

    start_sk = time.perf_counter()

    model = KNeighborsClassifier(n_neighbors=k) 
    model.fit(x_base, y_base)
    pred = model.predict(x_test)

    end_sk = time.perf_counter()

    sk_accu = 0
    for i in range(len(y_test)):
            if pred[i] == y_test[i]:
                sk_accu += 1

    cm = confusion_matrix(y_test, pred, labels=lenclass)

    if cross_valid == False:
        print("\n--- Результаты Sklearn ---")
        for i in range(len(y_test)):
            print(f"Класс в данных: {y_test[i]} -- Sklearn: {pred[i]}")
        print(f"Время работы Sklearn (одиночный тест): {end_sk - start_sk:.6f} сек.")
        print(f"Точность Sklearn: {(sk_accu/len(y_test))*100}%")
        draw_matrix(cm, "Одиночный тест: Sklearn", plt.cm.Blues)

        

    return sk_accu, cm

# Функция кросс-валидации
def cross_valid(n):
    global lenclass
    caccu = 0
    csk_accu = 0
    total_test = 0

    cross_matrix = [[0 for _ in lenclass] for _ in lenclass]
    sk_cross_cm = np.zeros((len(lenclass), len(lenclass)), dtype = int)

    for _ in range(n):
        base, test, data_set = separation(file)
        total_test += len(test)

        for i in range(len(test)):
            res = definition(test[i], data_set, k, True)

            xa1 = lenclass.index(test[i][3])
            xi1 = lenclass.index(res)
            cross_matrix[xa1][xi1] += 1

            if res == test[i][3]:
                caccu += 1
        sk_accu, cm = sklearn(base, test, k, True)
        csk_accu += sk_accu
        sk_cross_cm += cm
    
    print(f"\n--- Итоги Кросс-валидации ({n} итераций) ---")
    print(f"Средняя точность Алгоритма: {(caccu / total_test) * 100:.2f}%")
    print(f"Средняя точность Sklearn: {(csk_accu / total_test) * 100:.2f}%")

    draw_matrix(cross_matrix, f"Кросс-валидация ({n} ит.): Алгоритм", plt.cm.Greens)
    draw_matrix(sk_cross_cm, f"Кросс-валидация ({n} ит.): Sklearn", plt.cm.Oranges)

# матрицa ошибок
def draw_matrix(matrix_data, title_text, color_map):
    matrix_np = np.array(matrix_data)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix_np, display_labels=lenclass)
    
    disp.plot(cmap=color_map)
    
    plt.title(title_text)
    plt.show()

base, test, data_set = separation(file)

visualize_points(base, test)

print("\n--- Результаты Алгоритма ---")

start_alg = time.perf_counter()
for i in range(len(test)):
    definition(test[i], data_set ,k)
end_alg = time.perf_counter()

print(f"Время работы Алгоритма (одиночный тест): {end_alg - start_alg:.6f} сек.")
print(f"Точность алг: {(accu/len(test))* 100}%")

draw_matrix(matrix, "Одиночный тест: Алгоритм", plt.cm.Purples)

sklearn(base, test, k)
cross_valid(1000)
