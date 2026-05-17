import math
import random
import time
from sklearn.cluster import KMeans

#k = int(input("Введите кол-во кластеров:"))

file = open("citys.txt", "r", encoding="utf8")

# Создание массива с данными [город, x, y] и массива [x, y]
b = []
b1 = []

for i in file:
    i1 = i.strip().split()
    b.append([i1[0], float(i1[1]), float(i1[2])])
    b1.append([float(i1[1]), float(i1[2])])

# Создание матрицы расстояний
n = len(b)
dist = [[0] * n for _ in range(n)]

for i in range(n):
    for j in range(i + 1, n):
        c = math.sqrt((b[i][1] - b[j][1]) ** 2 + (b[i][2] - b[j][2]) ** 2)
        dist[i][j] = c
        dist[j][i] = c

#Метод Локтя
def elbow_method(b):
    all_wcss = []
    k = 1

    for i in range(1, 15):
        clusters, centroids = kmeans1(i, b, True)
        clwcss = calсwcss(clusters, centroids, b)
        all_wcss.append(clwcss)

    for i in range(1, len(all_wcss)):
        s = ((all_wcss[i-1] - all_wcss[i])/all_wcss[i-1])*100
        if s < 15:
            k = i
            break

    return k

# Алгоритмический
def algorithm(k, b, dist, wcss=False):
    start = time.perf_counter()
    n = len(b)
    centroids = [0]

    # Выбор центроидов
    while len(centroids) < k:
        maxdist = 0
        city = 0

        for i in range(n):
            if i in centroids: continue
            dista = min(dist[i][c] for c in centroids)
            if dista > maxdist:
                maxdist = dista
                city = i

        centroids.append(city)

    clustres = {}
    clustersid = {}

    for i in centroids:
        clustres[i] = []
        clustersid[i] = []

    # Заполнение кластеров по принципу отнесения каждого города к ближайшему центроиду
    for i in range(n):
        city_name = b[i][0]
        min_dist = float("inf")
        namecit = 0

        for ceni in centroids:
            cendist = dist[i][ceni]
            if cendist < min_dist:
                min_dist = cendist
                namecit = ceni

        clustres[namecit].append(city_name)
        clustersid[namecit].append(i)

    end = time.perf_counter()

    if wcss == True:
        centroidscord = []
        clusters_id = {}

        for i in range(len(centroids)):
            ka = centroids[i]
            centroidscord.append([b[ka][1], b[ka][2]])
            clusters_id[i] = clustersid[ka]

        return clusters_id, centroidscord

    else:
        #print('-----Алгоритмический-----')
        print(f"Время 'Алгоритмический': {end-start:.5f} сек")
        #print("\nКластеры:")
        #for i in range(1, k + 1):
        #print(f"Кластер {i}: {clustres[centroids[i-1]]}")

# KMeans
def kmeans(k, b1, b, wcss=False):
    start = time.perf_counter()
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(b1)
    centroids = kmeans.cluster_centers_.tolist()
    end = time.perf_counter()

    # 3. Формируем словарь кластеров с названиями городов и кластер с индексами
    clustres = {}
    clustersid = {}

    for i in range(k):
        clustres[i] = []
        clustersid[i] = []

    # Индексы городов по кластерам
    for i in range(len(labels)):
        idy = labels[i]
        clustersid[idy].append(i)

    # Названия городов по кластерам
    for i in range(len(b)):
        cluster_id = labels[i]
        city_name = b[i][0]
        clustres[cluster_id].append(city_name)

    if wcss == True:
        return clustersid, centroids
    else:
        #print("-----KMeans-----")
        print(f"Время 'KMeans': {end-start:.5f} сек")
        #print("\nКластеры:")
        #for c in clustres:
        #print(f"Кластер {c+1}: {clustres[c]}")

# Ручной KMeans
def kmeans1 (k, b, wcss=False):
    start = time.perf_counter()
    n = len(b)
    centroidsid = random.sample(range(n), k)
    centroids = []

    for i in centroidsid:
        centroids.append([b[i][1], b[i][2]])

    while True:
        clusters = {}

        for i in range(k):
            clusters[i] = []

        # Заполнение кластеров по принципу отнесения каждого города к ближайшему центроиду
        for i in range(n):
            city_x = b[i][1]
            city_y = b[i][2]
            min_dist = float("inf")
            icit = 0

            for cen in range(k):
                cen_x = centroids[cen][0]
                cen_y = centroids[cen][1]
                dista = math.sqrt((city_x - cen_x) ** 2 + (city_y - cen_y) ** 2)
                if dista < min_dist:
                    min_dist = dista
                    icit = cen

            clusters[icit].append(i)

        new_centroids = []

        # Новые центроиды
        for key in range(k):
            items = clusters[key]
            sum_x = 0
            sum_y = 0

            for item in items:
                sum_x += b[item][1]
                sum_y += b[item][2]

            mid_x = sum_x / len(items)
            mid_y = sum_y / len(items)
            new_centroids.append([mid_x, mid_y])

        if centroids == new_centroids:
            break

        centroids = new_centroids

    name_clusters = {}

    for key, items in clusters.items():
        name_list = []

        for i in items:
            name = b[i][0]
            name_list.append(name)

        name_clusters[key] = name_list

    end = time.perf_counter()

    if wcss == True:
        return clusters, centroids
    else:
        #print("-----Ручной KMeans-----")
        print(f"Время 'Ручной KMeans': {end-start:.5f} сек")
        #print("\nКластеры:")
        #for c in name_clusters:
        #print(f"Кластер {c+1}: {name_clusters[c]}")

#Wcss
def calсwcss(clusters, centroids, b):
    wcss = 0

    for i in range(len(centroids)):
        cen_x = centroids[i][0]
        cen_y = centroids[i][1]
        citys = clusters[i]

        for id in citys:
            city_x = b[id][1]
            city_y = b[id][2]
            dist = (city_x - cen_x)**2 + (city_y - cen_y)**2
            wcss += dist

    return wcss

k = elbow_method(b)

print(f"\n===Статистика (K = {k})===")

print("---Алгоритмический:")
aa1, bb1 = algorithm(k, b, dist, True)
alg1 = calсwcss(aa1, bb1, b)
print(f"WCSS: {round(alg1, 2)}")
algorithm(k, b, dist)

print("\n---Ручной KMeans:")
aa2, bb2 = kmeans1(k, b, True)
alg2 = calсwcss(aa2, bb2, b)
print(f"WCSS: {round(alg2, 2)}")
kmeans1(2, b)

print("\n---KMeans:")
aa3, bb3 = kmeans(k, b1, b, True)
alg3 = calсwcss(aa3, bb3, b)
print(f"WCSS: {round(alg3, 2)}")
kmeans(k, b1, b)