import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')

# Цветовая палитра
wine_red = '#8B0000'
pale_yellow_green = '#E8F4B7'
purple = '#6A0DAD'
light_purple = '#9370DB'

# Создание градиентов
red_gradient = LinearSegmentedColormap.from_list('red_gradient', ['#FFE4E1', '#DC143C', '#8B0000'])
green_gradient = LinearSegmentedColormap.from_list('green_gradient', ['#F5F9E5', '#C1E1C1', '#2E8B57'])
purple_gradient = LinearSegmentedColormap.from_list('purple_gradient', ['#E6E6FA', '#9370DB', '#4B0082'])

# 1. Создание матриц (данные с лекции)
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

print("=" * 60)
print("ДИСКРИМИНАНТНЫЙ АНАЛИЗ ПРЕДПРИЯТИЙ")
print("=" * 60)

# 2. Средние значения
sumX1 = x1.mean(axis=0)  # Среднее по столбцам для передовых предприятий
sumX2 = x2.mean(axis=0)  # Среднее по столбцам для отстающих предприятий
sumX0 = x0.mean(axis=0)  # Среднее по столбцам для классифицируемых предприятий

print("\n1. СРЕДНИЕ ЗНАЧЕНИЯ:")
print("Передовые предприятия (x1):", np.round(sumX1, 3))
print("Отстающие предприятия (x2):", np.round(sumX2, 3))
print("Классифицируемые предприятия (x0):", np.round(sumX0, 3))

# 3. Ковариационные матрицы
print("\n2. КОВАРИАЦИОННЫЕ МАТРИЦЫ:")

# Ручной расчет ковариационной матрицы для x1
s1 = np.zeros((3, 3)) 
for j in range(3):  # Проходим по всем столбцам (признакам)
    for l in range(3):  # Проходим по всем столбцам (признакам)
        s = 0
        for i in range(4):  # Проходим по всем строкам (предприятиям)
            s += (x1[i][j] - sumX1[j]) * (x1[i][l] - sumX1[l])
        s1[j][l] = s / 4  # Делим на количество наблюдений
print("Ковариационная матрица x1 (передовые):")
print(np.round(s1, 3))

# Ручной расчет ковариационной матрицы для x2
s2 = np.zeros((3, 3)) 
for j in range(3):  # Проходим по всем столбцам (признакам)
    for l in range(3):  # Проходим по всем столбцам (признакам)
        s = 0
        for i in range(5):  # Проходим по всем строкам (предприятиям)
            s += (x2[i][j] - sumX2[j]) * (x2[i][l] - sumX2[l])
        s2[j][l] = s / 5  # Делим на количество наблюдений
print("\nКовариационная матрица x2 (отстающие):")
print(np.round(s2, 3))

# Размеры выборок
n1 = len(x1)  # Количество передовых предприятий
n2 = len(x2)  # Количество отстающих предприятий
n0 = len(x0)  # Количество классифицируемых предприятий

# 4. Общая ковариационная матрица
S_all = 1/(n1 + n2 - 2) * (n1 * s1 + n2 * s2)  # Взвешенное среднее ковариационных матриц
print("\n3. ОБЩАЯ КОВАРИАЦИОННАЯ МАТРИЦА:")
print(np.round(S_all, 3))

# 5. Обратная матрица
SO = np.linalg.inv(S_all)  # Вычисление обратной матрицы
print("\n4. ОБРАТНАЯ МАТРИЦА:")
print(np.round(SO, 3))

# 6. Дискриминантная переменная (коэффициенты дискриминантной функции)
A = np.sum(SO * (sumX1 - sumX2), axis=1)  # Вычисление коэффициентов дискриминантной функции
print("\n5. ДИСКРИМИНАНТНАЯ ПЕРЕМЕННАЯ (коэффициенты A):")
print(np.round(A, 6))

# 7. Вычисление дискриминантных значений F
F1 = np.sum(A * x1, axis=1)  # Дискриминантные значения для передовых предприятий
F2 = np.sum(A * x2, axis=1)  # Дискриминантные значения для отстающих предприятий

print("\n6. ДИСКРИМИНАНТНЫЕ ЗНАЧЕНИЯ:")
print("F1 (передовые):", np.round(F1, 3))
print("F2 (отстающие):", np.round(F2, 3))

# 8. Средние дискриминантные значения
F1_mean = F1.mean(axis=0)  # Среднее значение F для передовых предприятий
F2_mean = F2.mean(axis=0)  # Среднее значение F для отстающих предприятий

print("\n7. СРЕДНИЕ ДИСКРИМИНАНТНЫЕ ЗНАЧЕНИЯ:")
print(f"F1_mean (передовые): {F1_mean:.3f}")
print(f"F2_mean (отстающие): {F2_mean:.3f}")

# 9. Общее пороговое значение F
F_all = 1/2 * (F1_mean + F2_mean)  # Среднее между двумя группами
print(f"\n8. ОБЩЕЕ ПОРОГОВОЕ ЗНАЧЕНИЕ F: {F_all:.3f}")

# 10. Дискриминантные значения для классифицируемых предприятий
F0 = np.sum(A * x0, axis=1)  # Дискриминантные значения для тестовых предприятий
print(f"\n9. ДИСКРИМИНАНТНЫЕ ЗНАЧЕНИЯ ДЛЯ КЛАССИФИЦИРУЕМЫХ ПРЕДПРИЯТИЙ:")
print("F0:", np.round(F0, 3))

# 11. Разница от порогового значения
raz = F0 - F_all  # Отклонение от порогового значения
print(f"\n10. ОТКЛОНЕНИЕ ОТ ПОРОГА:")
print("Разница:", np.round(raz, 3))

# 12. Классификация предприятий
print("\n11. РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:")
print("=" * 50)

for i in range(len(F0)):
    if F1_mean > F2_mean: 
        if raz[i] > 0:
            result = "ПЕРЕДОВОЕ"
            color = wine_red
        else: 
            result = "ОТСТАЮЩЕЕ"
            color = purple
    else:
        if raz[i] < 0:
            result = "ПЕРЕДОВОЕ"
            color = wine_red
        else: 
            result = "ОТСТАЮЩЕЕ"
            color = purple
    
    print(f"Предприятие {i+1}: F0 = {F0[i]:.3f}, Отклонение = {raz[i]:.3f} → {result}")

# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
print("\n12. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 50)

# Создание фигуры с несколькими графиками
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Дискриминантные значения всех предприятий
all_F = np.concatenate([F1, F2, F0])  # Все дискриминантные значения
all_labels = (['Передовые'] * len(F1) + ['Отстающие'] * len(F2) + 
             ['Классифицируемые'] * len(F0))  # Метки групп
all_colors = [wine_red] * len(F1) + [purple] * len(F2) + [pale_yellow_green] * len(F0)  # Цвета групп

# Scatter plot дискриминантных значений
scatter = ax1.scatter(range(len(all_F)), all_F, c=all_colors, s=100, alpha=0.7, edgecolors='black')
ax1.axhline(y=F_all, color='red', linestyle='--', linewidth=2, label=f'Порог F = {F_all:.3f}')
ax1.set_xlabel('Номер предприятия')
ax1.set_ylabel('Дискриминантное значение F')
ax1.set_title('Дискриминантные значения предприятий', fontsize=14, fontweight='bold', color=purple)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Создание легенды
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=wine_red, markersize=8, label='Передовые'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=purple, markersize=8, label='Отстающие'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pale_yellow_green, markersize=8, label='Классифицируемые')
]
ax1.legend(handles=legend_elements)

# График 2: Распределение по признакам (3D проекция)
feature_names = ['Признак 1', 'Признак 2', 'Признак 3']
for i, feature in enumerate(feature_names):
    # Объединяем данные по каждому признаку
    feature_data = np.concatenate([x1[:, i], x2[:, i], x0[:, i]])
    feature_labels = np.concatenate([
        np.full(len(F1), 1),  # Передовые
        np.full(len(F2), 2),  # Отстающие  
        np.full(len(F0), 3)   # Классифицируемые
    ])
    
    # Boxplot по группам
    data_by_group = [x1[:, i], x2[:, i], x0[:, i]]
    box_plot = ax2.boxplot(data_by_group, labels=['Передовые', 'Отстающие', 'Классифицируемые'], 
                          patch_artist=True)
    
    # Настройка цветов boxplot
    colors_box = [wine_red, purple, pale_yellow_green]
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

ax2.set_ylabel('Значения признаков')
ax2.set_title('Распределение значений признаков по группам', fontsize=14, fontweight='bold', color=purple)
ax2.grid(True, alpha=0.3)

# График 3: Отклонения от порога для классифицируемых предприятий
bars = ax3.bar(range(len(raz)), raz, color=[wine_red if r > 0 else purple for r in raz], alpha=0.7)
ax3.axhline(y=0, color='black', linewidth=1)
ax3.set_xlabel('Классифицируемые предприятия')
ax3.set_ylabel('Отклонение от порога')
ax3.set_title('Отклонения дискриминантных значений от порога', fontsize=14, fontweight='bold', color=purple)
ax3.set_xticks(range(len(raz)))
ax3.set_xticklabels([f'Предпр. {i+1}' for i in range(len(raz))])
ax3.grid(True, alpha=0.3)

# Добавление значений на столбцы
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{raz[i]:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

# График 4: Сравнение средних значений признаков
features_mean = np.vstack([sumX1, sumX2, sumX0])
x_pos = np.arange(len(feature_names))
width = 0.25

bars1 = ax4.bar(x_pos - width, features_mean[0], width, label='Передовые', color=wine_red, alpha=0.7)
bars2 = ax4.bar(x_pos, features_mean[1], width, label='Отстающие', color=purple, alpha=0.7)  
bars3 = ax4.bar(x_pos + width, features_mean[2], width, label='Классифицируемые', color=pale_yellow_green, alpha=0.7)

ax4.set_xlabel('Признаки')
ax4.set_ylabel('Средние значения')
ax4.set_title('Сравнение средних значений признаков', fontsize=14, fontweight='bold', color=purple)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(feature_names)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ДОПОЛНИТЕЛЬНО: Анализ с датасетом вина
print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ: ДАТАСЕТ ВИНА")
print("=" * 60)

try:
    # Загрузка датасета вина
    wine_data = pd.read_csv('winequality-white.csv', delimiter=';', nrows=50)
    print(f"Загружен датасет вина: {wine_data.shape}")
    
    # Выбираем признаки для анализа (первые 3 числовых признака)
    numeric_columns = wine_data.select_dtypes(include=[np.number]).columns[:3]
    wine_features = wine_data[numeric_columns].values
    
    print(f"Используемые признаки: {list(numeric_columns)}")
    print(f"Размерность данных: {wine_features.shape}")
    
    # Делим данные на две группы (условно) на основе медианы первого признака
    median_val = np.median(wine_features[:, 0])
    group1_mask = wine_features[:, 0] > median_val
    group2_mask = ~group1_mask
    
    wine_x1 = wine_features[group1_mask]
    wine_x2 = wine_features[group2_mask]
    
    print(f"\nГруппа 1 (высокие значения): {len(wine_x1)} образцов")
    print(f"Группа 2 (низкие значения): {len(wine_x2)} образцов")
    
    # Применяем дискриминантный анализ к данным вина
    if len(wine_x1) > 1 and len(wine_x2) > 1:
        wine_sumX1 = wine_x1.mean(axis=0)
        wine_sumX2 = wine_x2.mean(axis=0)
        
        print(f"\nСредние значения для вина:")
        print(f"Группа 1: {np.round(wine_sumX1, 3)}")
        print(f"Группа 2: {np.round(wine_sumX2, 3)}")
        
        # Упрощенный анализ для демонстрации
        print("\nДискриминантный анализ может быть применен к данным вина")
        print("для классификации по качеству или другим характеристикам")
        
except FileNotFoundError:
    print("Файл winequality-white.csv не найден.")
    print("Продолжаем анализ только с основными данными предприятий.")

print("\n" + "=" * 60)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 60)