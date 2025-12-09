import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

def lda_algorithm(x1, x2, x0, class1_name="Класс 1", class2_name="Класс 2"):
    
    # 2. Средние векторы
    sumX1 = x1.mean(axis=0)
    sumX2 = x2.mean(axis=0)
    sumX0 = x0.mean(axis=0)
    print("Средний вектор класса 1:\n", sumX1)
    print("\nСредний вектор класса 2:\n", sumX2)
    print("\nСредний вектор тестовых объектов:\n", sumX0)

    # 3. Ковариационные матрицы
    s1 = np.zeros((x1.shape[1], x1.shape[1])) 
    for j in range(x1.shape[1]):  
        for l in range(x1.shape[1]):  
            s = 0
            for i in range(len(x1)):  
                s += (x1[i][j] - sumX1[j]) * (x1[i][l] - sumX1[l])
            s1[j][l] = s / len(x1)  
    print("Ковариационная матрица класса 1:")
    print(s1)

    s2 = np.zeros((x2.shape[1], x2.shape[1])) 
    for j in range(x2.shape[1]):  
        for l in range(x2.shape[1]):  
            s = 0
            for i in range(len(x2)):  
                s += (x2[i][j] - sumX2[j]) * (x2[i][l] - sumX2[l])
            s2[j][l] = s / len(x2)  
    print("Ковариационная матрица класса 2:")
    print(s2)

    n1 = len(x1)
    n2 = len(x2)
    
    # 4. Общая ковариационная матрица
    S_all = 1/(n1+n2-2)*(n1*s1+n2*s2)
    print("Общая ковариационная матрица:")
    print(S_all)

    # 5. Обратная матрица
    SO = np.linalg.inv(S_all)
    print("Обратная матрица:")
    print(SO)

    # 6. Дискриминантная переменная
    A = np.sum(SO*(sumX1-sumX2), axis=1)
    print("Коэффициенты дискриминантной функции:")
    print(A)

    # 7. Находим F
    F1 = np.sum(A*x1, axis=1)
    print("F для класса 1: ")
    print(F1)

    F2 = np.sum(A*x2, axis=1)
    print("F для класса 2: ")
    print(F2)

    # 8. Среднее F
    F1_mean = F1.mean(axis=0)
    print("Среднее F для класса 1:")
    print(F1_mean)

    F2_mean = F2.mean(axis=0)
    print("Среднее F для класса 2:")
    print(F2_mean)

    # 9. Общее F (порог)
    F_all = 1/2*(F1_mean+F2_mean)
    print(f"Пороговое значение F: \n {F_all}")

    # 10. F для тестовых объектов
    F0 = np.sum(A*x0, axis=1)
    print(f"F для тестовых объектов: \n {F0}")

    # 11 Разница от порога
    raz = F0 - F_all
    print(f"Разница от порога: \n {raz}")

    # 12. Классификация
    classifications = []
    print(f"\nРезультаты классификации:")
    for i in range(len(F0)):
        if F1_mean > F2_mean: 
            if raz[i] > 0:
                classification = class1_name
            else: 
                classification = class2_name
        else:
            if raz[i] < 0:
                classification = class1_name
            else: 
                classification = class2_name
        classifications.append(classification)
        print(f"Объект {i+1}: F = {F0[i]:.4f} → {classification}")

    return F1, F2, F0, F_all, classifications

def plot_results(F1, F2, F0, F_all, title, class1_name="Класс 1", class2_name="Класс 2"):
    plt.figure(figsize=(12, 6))
    
    indices_F1 = range(len(F1))                                    
    indices_F2 = range(len(F1), len(F1) + len(F2))                 
    indices_F0 = range(len(F1) + len(F2), len(F1) + len(F2) + len(F0)) 
    
    plt.scatter(indices_F1, F1, label=class1_name, marker='x', s=100, linewidth=2, color='blue')
    plt.scatter(indices_F2, F2, label=class2_name, marker='x', s=100, linewidth=2, color='red')
    
    plt.scatter(indices_F0, F0, label='Тестовые объекты', marker='o', s=100, color='green')
    
    # Линия границы
    plt.axhline(F_all, color='black', linewidth=2, linestyle='--', 
                label=f'Граница F = {F_all:.4f}')
    
    plt.xlabel("Номер объекта")
    plt.ylabel("Дискриминантная переменная F")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_original_example():
   
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ НА ОРИГИНАЛЬНОМ ПРИМЕРЕ")
    print("=" * 60)
    
    # 1. Создание матриц
    x1 = np.array([
        [224.228, 17.115, 22.981],
        [151.827, 14.904, 21.481],
        [147.313, 13.627, 18.669],
        [152.253, 10.545, 10.199]
    ])

    x2 = np.array([
        [46.757, 4.428, 11.124],
        [29.033, 5.510, 6.091],
        [52.134, 4.214, 11.842],
        [37.050, 5.527, 11.873],
        [63.979, 4.211, 12.860]
    ])

    x0 = np.array([
        [55.451, 9.592, 12.840],
        [78.575, 11.727, 15.535],
        [98.353, 17.572, 20.458]
    ])
    
    F1, F2, F0, F_all, classifications = lda_algorithm(
        x1, x2, x0, 
        class1_name="Передовые предприятия", 
        class2_name="Отстающие предприятия"
    )
    
    plot_results(
        F1, F2, F0, F_all, 
        "LDA: Оригинальный пример с предприятиями",
        class1_name="Передовые предприятия",
        class2_name="Отстающие предприятия"
    )

def test_seeds_dataset():
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НА ДАТАСЕТЕ SEEDS")
    print("=" * 60)
    
    seeds = fetch_openml(name='seeds', version=1, as_frame=False)
    X = seeds.data
    y = seeds.target
    
    print(f"Размер датасета: {X.shape}")
    print(f"Классы: {np.unique(y)}")
    
    class1_mask = (y == '1')
    class2_mask = (y == '2')
    
    x1 = X[class1_mask]
    x2 = X[class2_mask]
    
    print(f"Класс 1: {len(x1)} объектов")
    print(f"Класс 2: {len(x2)} объектов")
    
    train_size1 = int(0.7 * len(x1))
    train_size2 = int(0.7 * len(x2))
    
    x1_train = x1[:train_size1]
    x1_test = x1[train_size1:]
    
    x2_train = x2[:train_size2]
    x2_test = x2[train_size2:]
    
    x0 = np.vstack([x1_test, x2_test])
    
    print(f"\nОбучающая выборка:")
    print(f"Класс 1: {len(x1_train)} объектов")
    print(f"Класс 2: {len(x2_train)} объектов")
    print(f"Тестовая выборка: {len(x0)} объектов")
    
    F1, F2, F0, F_all, classifications = lda_algorithm(
        x1_train, x2_train, x0,
        class1_name="Тип семян 1", 
        class2_name="Тип семян 2"
    )
    
    plot_results(
        F1, F2, F0, F_all, 
        "LDA: Датсет Seeds (первые два класса)",
        class1_name="Тип семян 1",
        class2_name="Тип семян 2"
    )
    
    # Анализ качества классификации
    true_labels = ['Тип семян 1'] * len(x1_test) + ['Тип семян 2'] * len(x2_test)
    correct = sum(1 for i in range(len(classifications)) if classifications[i] == true_labels[i])
    accuracy = correct / len(classifications) * 100
    
    print(f"\nТочность классификации на тестовой выборке: {accuracy:.2f}%")

if __name__ == "__main__":
    test_original_example()
    test_seeds_dataset()
    