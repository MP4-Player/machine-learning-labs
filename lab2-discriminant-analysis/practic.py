import numpy as np
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

# 2. Среднии суммы
sumX1 = x1.mean(axis=0)
sumX2 = x2.mean(axis=0)
sumX0 = x0.mean(axis=0)
print("Матрица передовые предприятия:\n", sumX1)
print("\nМатрица отстающие предприятия:\n", sumX2)
print("\nМатрица предприятия для классификации:\n", sumX0)

# 3. Ковариационные матрицы

cov_x1 = np.cov(x1, rowvar=False)

s1 = np.zeros((3, 3)) 
for j in range(3):  
    for l in range(3):  
        s = 0
        for i in range(4):  
            s += (x1[i][j] - sumX1[j]) * (x1[i][l] - sumX1[l])
        s1[j][l] = s / 4  
print("Ковариационная матрица x1:")
print(s1)

s2 = np.zeros((3, 3)) 
for j in range(3):  
    for l in range(3):  
        s = 0
        for i in range(5):  
            s += (x2[i][j] - sumX2[j]) * (x2[i][l] - sumX2[l])
        s2[j][l] = s / 5  
print("Ковариационная матрица x2:")
print(s2)

# cov_x1 = np.cov(x1, rowvar=False)
# print("Ковариационная матрица x1:")
# print(cov_x1)
n1=len(x1)
n2=len(x2)
n3=len(x0)
# 4. Общая ковариационная
S_all = 1/(n1+n2-2)*(n1*s1+n2*s2)
print("Общая ковариационная")
print(S_all)

# 5.Обратная матрица
SO=np.linalg.inv(S_all)
print("Обратная матрица")
print(SO)

# 6. Дискриминантная переменная
A=np.sum(SO*(sumX1-sumX2), axis=1)
print("Дискриминантная переменная")
print(A)

# 7. Находим F
F1=np.sum(A*x1, axis=1)
print("F1: ")
print(F1)

F2=np.sum(A*x2, axis=1)
print("F2: ")
print(F2)

# 8. Среднее F
F1_mean=F1.mean(axis=0)
print("F1_mean")
print(F1_mean)

F2_mean=F2.mean(axis=0)
print("F2_mean")
print(F2_mean)

# 9. Общее F
F_all=1/2*(F1_mean+F2_mean)
print(f"Общее F: \n {F_all}")

# 10. F для x0
F0=np.sum(A*x0, axis=1)
print(f"F0: \n {F0}")

# 11 Разница
raz=F0-F_all
print(f"Разница: \n {raz}")

# 12. Выводы
if F1_mean > F2_mean: 
    if raz[0] > 0:
        print(f"Объект относится к множеству x1 - передовых")
    else: 
        print(f"Объект относится к множеству x2 - отстающих")
else:
    if raz[0] < 0:
        print(f"Объект относится к множеству x1 - передовых")
    else: 
        print(f"Объект относится к множеству x2 - отстающих")
