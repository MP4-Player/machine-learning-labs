import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Настройка вывода для лучшей читаемости
np.set_printoptions(precision=2, suppress=True)
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# Чтение данных
df = pd.read_csv(r'C:\Users\Пользователь\Desktop\програмироание\mecanicoratrismeckanikkavenal\lab-5\primer.csv', encoding='utf-8')

x = df['Доходы_руб'].values.astype(float)  # Вектор независимой переменной (доходы)
y = df['Расходы_руб'].values.astype(float)  # Вектор зависимой переменной (расходы)
n = len(x)  # Размер выборки

# Формула 0: Средние значения
x_mean = np.mean(x)
y_mean = np.mean(y)

print("=" * 70)
print("РЕГРЕССИОННЫЙ АНАЛИЗ: Зависимость расходов от доходов")
print("=" * 70)

# ========== 1. ПАРАМЕТРЫ РЕГРЕССИИ ==========
print("\n1. ПОСТРОЕНИЕ МОДЕЛИ РЕГРЕССИИ")
print("-" * 70)

sum_xy = np.sum((x - x_mean) * (y - y_mean))
sum_xx = np.sum((x - x_mean)**2)

print(f"Среднее значение доходов (x̄): {x_mean:.2f} руб.")
print(f"Среднее значение расходов (ȳ): {y_mean:.2f} руб.")
print(f"∑(xᵢ - x̄)(yᵢ - ȳ) = {sum_xy:.2f}")
print(f"∑(xᵢ - x̄)² = {sum_xx:.2f}")

b = sum_xy / sum_xx
a = y_mean - b * x_mean

print(f"\nПараметры модели:")
print(f"  β̂ (коэффициент наклона) = {b:.2f}")
print(f"  â (свободный член) = {a:.2f}")
print(f"\nМодель регрессии: ŷ = {a:.2f} + {b:.2f}·x")
print(f"Интерпретация: При увеличении дохода на 1 руб. расходы увеличиваются в среднем на {b:.2f} руб.")

X = np.column_stack([np.ones(n), x])
Y = y.reshape(-1, 1)
XTX_inv = np.linalg.inv(X.T @ X)
params_matrix = XTX_inv @ X.T @ Y
print(f"\nПроверка (матричная форма):")
print(f"  â = {params_matrix[0,0]:.2f}, β̂ = {params_matrix[1,0]:.2f}")

y_pred = a + b * x
ss_res = np.sum((y - y_pred)**2)
s2 = ss_res / (n - 2)
S_e = np.sqrt(s2)

print(f"\nОстаточная дисперсия:")
print(f"  ∑eᵢ² = {ss_res:.2f}")
print(f"  s² = ∑eᵢ²/(n-2) = {ss_res:.2f}/{n-2} = {s2:.2f}")
print(f"  S_e = √s² = {S_e:.2f}")

# ========== 2. ПРОВЕРКА ЗНАЧИМОСТИ ПАРАМЕТРОВ (α = 0.1) ==========
print("\n" + "=" * 70)
print("2. ПРОВЕРКА ЗНАЧИМОСТИ ПАРАМЕТРОВ МОДЕЛИ (α = 0.1)")
print("-" * 70)

s_b = np.sqrt(s2 / sum_xx)
t_b = b / s_b
t_crit_01 = stats.t.ppf(1 - 0.1/2, n-2)

print(f"\nПараметр β (коэффициент наклона):")
print(f"  S_b = √(s²/∑(xᵢ - x̄)²) = √({s2:.2f}/{sum_xx:.2f}) = {s_b:.4f}")
print(f"  t_расч = |β̂|/S_b = {b:.2f}/{s_b:.4f} = {t_b:.2f}")
print(f"  t_табл (α=0.1, df={n-2}) = {t_crit_01:.2f}")
if abs(t_b) > t_crit_01:
    print(f"  ✓ |t_расч| > t_табл → Параметр β ЗНАЧИМ на уровне 0.1")
else:
    print(f"  ✗ |t_расч| ≤ t_табл → Параметр β НЕ ЗНАЧИМ")

s_a = np.sqrt(s2 * (1/n + x_mean**2 / sum_xx))
t_a = a / s_a

print(f"\nПараметр α (свободный член):")
print(f"  S_a = √(s²(1/n + x̄²/∑(xᵢ - x̄)²)) = {s_a:.4f}")
print(f"  t_расч = |â|/S_a = {a:.2f}/{s_a:.4f} = {t_a:.2f}")
print(f"  t_табл (α=0.1, df={n-2}) = {t_crit_01:.2f}")
if abs(t_a) > t_crit_01:
    print(f"  ✓ |t_расч| > t_табл → Параметр α ЗНАЧИМ на уровне 0.1")
else:
    print(f"  ✗ |t_расч| ≤ t_табл → Параметр α НЕ ЗНАЧИМ")

# ========== 3. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ДЛЯ МОДЕЛИ (α = 0.05) ==========
print("\n" + "=" * 70)
print("3. ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ ДЛЯ МОДЕЛИ РЕГРЕССИИ (α = 0.05)")
print("-" * 70)

sort_idx = np.argsort(x)
x_sorted = x[sort_idx]
y_sorted = y[sort_idx]

alpha_05 = 0.05
t_crit_05 = stats.t.ppf(1 - alpha_05/2, n-2)

print(f"\nКоэффициент Стьюдента t_α (α=0.05, df={n-2}) = {t_crit_05:.2f}")

denom = sum_xx
se_pred = np.sqrt(s2 * (1 + 1/n + (x_sorted - x_mean)**2 / denom))
lower_band = a + b * x_sorted - t_crit_05 * se_pred
upper_band = a + b * x_sorted + t_crit_05 * se_pred

print(f"Формула: U(α=0.05) = S_e · t_α · √(1 + 1/n + (xᵢ - x̄)²/∑(xⱼ - x̄)²)")
print(f"где S_e = {S_e:.2f}, t_α = {t_crit_05:.2f}")

# ============ ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ДЛЯ ГРАДИЕНТА ============
def gradient_fill_between(ax, x, y1, y2, colors=['pink', 'lightblue', 'yellow'], alpha=0.35):
    """
    Заливка между y1 и y2 с плавным градиентом по X.
    Цвет зависит от нормализованной позиции x (от 0 до 1).
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap

    # Создаём colormap
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    # Нормализуем x в диапазон [0, 1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)

    # Формируем полигоны: для каждой соседней пары точек — один четырёхугольник
    verts = []
    facecolors = []

    for i in range(len(x) - 1):
        # Координаты четырёхугольника: нижняя → верхняя → вперёд → назад
        poly = [
            (x[i],     y1[i]),
            (x[i],     y2[i]),
            (x[i + 1], y2[i + 1]),
            (x[i + 1], y1[i + 1]),
            (x[i],     y1[i])  # замыкаем
        ]
        verts.append(poly)

        # Цвет — по средней x-позиции сегмента
        x_mid = (x_norm[i] + x_norm[i + 1]) / 2
        facecolors.append(cmap(x_mid))

    # Создаём коллекцию полигонов
    poly_collection = PolyCollection(
        verts,
        facecolors=facecolors,
        alpha=alpha,
        edgecolor='none'
    )
    ax.add_collection(poly_collection)
    ax.autoscale_view()

# График 1: Исходные данные, модель и доверительные интервалы (α = 0.05)
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.scatter(x, y, color='black', s=60, label='Исходные данные', zorder=3)
ax1.plot(x_sorted, a + b * x_sorted, color='black', linewidth=2.5, 
         label=f'Линия регрессии: ŷ = {a:.2f} + {b:.2f}·x', zorder=2)

# === ГРАДИЕНТНАЯ ЗАЛИВКА ===
gradient_fill_between(ax1, x_sorted, lower_band, upper_band, alpha=0.35)

ax1.plot(x_sorted, lower_band, color='darkblue', linestyle='--', linewidth=1.2, 
         label='95% доверительные границы (α=0.05)', zorder=1)
ax1.plot(x_sorted, upper_band, color='darkblue', linestyle='--', linewidth=1.2, zorder=1)

ax1.set_xlabel('Доходы, руб.', fontsize=12)
ax1.set_ylabel('Расходы, руб.', fontsize=12)
ax1.set_title('Рис. 3.3.1. График исходных данных, модели и доверительных интервалов (α=0.05)', 
              fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1500, 3700)
ax1.set_ylim(1000, 3700)
plt.tight_layout()
plt.show()

# ========== 4. ПРОГНОЗ ПРИ x₀ = 3600 руб. ==========
print("\n" + "=" * 70)
print("4. ОЦЕНКА РАСХОДОВ ПРИ ДОХОДЕ 3600 руб.")
print("-" * 70)

x0 = 3600
y0 = a + b * x0

print(f"\nТочечный прогноз:")
print(f"  ŷ_прогн = {a:.2f} + {b:.2f} · {x0} = {y0:.2f} руб.")

alpha_01 = 0.1
t_crit_01_pred = stats.t.ppf(1 - alpha_01/2, n-2)
se_x0 = np.sqrt(s2 * (1 + 1/n + (x0 - x_mean)**2 / denom))
delta_01 = t_crit_01_pred * se_x0

lower_90 = y0 - delta_01
upper_90 = y0 + delta_01

print(f"\nДоверительный интервал (α = 0.1, т.е. 90%):")
print(f"  U(x=3600; n={n}; α=0.1) = S_e · t_α · √(1 + 1/n + (x₀ - x̄)²/∑(xⱼ - x̄)²)")
print(f"  U = {S_e:.2f} · {t_crit_01_pred:.2f} · √(1 + 1/{n} + ({x0} - {x_mean:.2f})²/{denom:.2f})")
print(f"  U = {delta_01:.2f}")
print(f"\n  Нижняя граница: {y0:.2f} - {delta_01:.2f} = {lower_90:.2f} руб.")
print(f"  Верхняя граница: {y0:.2f} + {delta_01:.2f} = {upper_90:.2f} руб.")
print(f"  Ширина интервала (2·Δ): {2*delta_01:.2f} руб.")
print(f"\n  Прогнозное значение ŷ_прогн = {y0:.2f} с вероятностью 90% находится в интервале:")
print(f"  [{lower_90:.2f}; {upper_90:.2f}] руб.")

# График 2: Детальный вид с прогнозом при x = 3600
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.scatter(x, y, color='black', s=60, label='Исходные данные', zorder=3)
ax2.plot(x_sorted, a + b * x_sorted, color='black', linewidth=2.5, 
         label=f'Линия регрессии: ŷ = {a:.2f} + {b:.2f}·x', zorder=2)

# === ГРАДИЕНТНАЯ ЗАЛИВКА ===
gradient_fill_between(ax2, x_sorted, lower_band, upper_band, alpha=0.25)

ax2.plot(x_sorted, lower_band, color='darkblue', linestyle='--', linewidth=1.2, 
         label='95% доверительные границы (α=0.05)', zorder=1)
ax2.plot(x_sorted, upper_band, color='darkblue', linestyle='--', linewidth=1.2, zorder=1)

# Точка прогноза
ax2.scatter(x0, y0, color='pink', s=150, marker='*', zorder=5, 
           label=f'Прогноз при x={x0}: {y0:.2f} руб.', edgecolors='darkred', linewidths=1.5)
ax2.errorbar(x0, y0, yerr=delta_01, fmt='none', ecolor='pink', capsize=8, 
            capthick=2, linewidth=2.5, zorder=4,
            label=f'90% интервал: [{lower_90:.0f}; {upper_90:.0f}] руб.')

ax2.axvline(x=x0, color='pink', linestyle=':', linewidth=1, alpha=0.5, zorder=0)

ax2.text(x0 + 80, upper_90, f'{upper_90:.2f}', fontsize=11, color='pink', 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='pink', alpha=0.8))
ax2.text(x0 + 80, y0, f'{y0:.2f}', fontsize=11, color='pink', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='pink', alpha=0.8))
ax2.text(x0 + 80, lower_90, f'{lower_90:.2f}', fontsize=11, color='pink',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='pink', alpha=0.8))

ax2.set_xlabel('Доходы, руб.', fontsize=12)
ax2.set_ylabel('Расходы, руб.', fontsize=12)
ax2.set_title('Рис. 3.3.2. График модели и интервал прогноза при x = 3600 руб.', 
              fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1500, 3800)
ax2.set_ylim(1000, 3800)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 70)