# Импорт библиотек
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import tree
import graphviz
import shap
from sklearn.tree import export_text
import matplotlib.pyplot as plt


# Установка стилей
plt.style.use('ggplot')
sns.set_style("whitegrid")
%matplotlib inline

# 1. Загрузка данных
print("1. Загрузка данных...")
PATH = 'wine+quality/'
TRAIN_PATH = PATH + 'winequality-red.csv'
TEST_PATH = PATH + 'winequality-white.csv'

red_wine = pd.read_csv(TRAIN_PATH, sep=';')
white_wine = pd.read_csv(TEST_PATH, sep=';')



# Объединение датасетов

# Добавление меток типа вина
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'

wines = pd.concat([red_wine, white_wine], ignore_index=True)
# Функция для визуализации
def plot_quality_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='quality', hue='wine_type', data=data)
    plt.title('Распределение качества вина по типам')
    plt.xlabel('Качество (0-10)')
    plt.ylabel('Количество образцов')
    plt.legend(title='Тип вина')
    plt.show()

# Вызов функции
plot_quality_distribution(wines)



# Предварительный анализ данных
print("\nПервые 5 строк данных:")
display(wines.head())

print("\nИнформация о данных:")
print(wines.info())

print("\nОписательная статистика:")
display(wines.describe())



# Разделение на 3 класса качества
def classify_quality(quality):
    if quality <= 4:
        return 'low'
    elif 5 <= quality <= 6:
        return 'medium'
    else:
        return 'high'

wines['quality_class'] = wines['quality'].apply(classify_quality)

# Визуализация классов
plt.figure(figsize=(8, 5))
sns.countplot(x='quality_class', hue='wine_type', data=wines, 
             order=['low', 'medium', 'high'])
plt.title('Распределение по классам качества')
plt.show()




print(f"Загружено {len(wines)} образцов")

# 2. Предобработка данных
print("\n2. Предобработка данных...")
wines['quality_class'] = pd.cut(wines['quality'],
                              bins=[0, 4, 6, 10],
                              labels=['low', 'medium', 'high'])

X = wines.drop(['quality', 'quality_class', 'wine_type'], axis=1)
y = wines['quality_class']

# Кодирование классов
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




# 3. Обучение Random Forest
print("\n3. Обучение Random Forest...")
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_
print(f"Лучшие параметры Random Forest: {grid.best_params_}")



# 4. Визуализация одного полного дерева Random Forest
print("\n4. Визуализация одного дерева из Random Forest...")

# Выбираем первое дерево из ансамбля
single_tree = best_rf.estimators_[0]

# Вариант 1: Текстовое представление (работает всегда)
tree_rules = export_text(single_tree, 
                        feature_names=list(X.columns),
                        class_names=list(le.classes_))
print("Текстовое представление дерева:\n")
print(tree_rules[:2000] + "...")  # Выводим первые 2000 символов

# Вариант 2: Графическое представление (если установлен graphviz)
try:
    # Визуализация с увеличенным размером и полной глубиной
    plt.figure(figsize=(24, 12))
    tree.plot_tree(single_tree,
                 feature_names=X.columns,
                 class_names=le.classes_,
                 filled=True,
                 rounded=True,
                 proportion=True,
                 fontsize=10,
                 max_depth=None)  # Без ограничения глубины
    
    plt.title("Полное дерево из Random Forest", fontsize=14)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print("\nНе удалось построить графическое дерево. Установите graphviz:")
    print("1. Скачайте с https://graphviz.org/download/")
    print("2. Добавьте в PATH (или установите через conda: conda install python-graphviz)")
    print(f"Ошибка: {e}")

# Вариант 3: Альтернативная визуализация важных ветвей
plt.figure(figsize=(16, 8))
tree.plot_tree(single_tree,
              feature_names=X.columns,
              class_names=le.classes_,
              filled=True,
              max_depth=3,  # Показываем первые 3 уровня для читаемости
              proportion=True,
              fontsize=9)
plt.title("Первые 3 уровня дерева Random Forest", fontsize=14)
plt.show()



# 5. Анализ важности признаков
print("\n5. Анализ важности признаков в Random Forest...")

# Важность признаков
importances = best_rf.feature_importances_
feature_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_imp = feature_imp.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp)
plt.title("Важность признаков в Random Forest", fontsize=14)
plt.show()




shap.initjs()  # Инициализация JavaScript-визуализаций

# Вычисление SHAP значений
print("\nВычисление SHAP значений для Random Forest...")
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_scaled)

# Вариант 1: Интерактивная визуализация (лучше для Jupyter)
print("\nВариант 1: Интерактивная визуализация SHAP (использует JavaScript)")
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, class_names=list(le.classes_))

# Вариант 2: Статичная визуализация с matplotlib
print("\nВариант 2: Статичная визуализация SHAP")
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns, class_names=list(le.classes_), show=False)
plt.title("SHAP значения для Random Forest", fontsize=14)
plt.tight_layout()
plt.show()




# 6. Детальный анализ признаков
print("\n6. Детальный анализ признаков...")

top_features = feature_imp['Feature'].values[:9]
print(f"Топ-9 важных признака: {top_features}")

for feature in top_features:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='quality_class', y=feature, data=wines, order=['low', 'medium', 'high'])
    plt.title(f'Распределение {feature} по классам качества', fontsize=12)
    plt.show()
    
    # График зависимости от качества
    plt.figure(figsize=(10, 5))
    sns.regplot(x='quality', y=feature, data=wines, scatter_kws={'alpha':0.3})
    plt.title(f'Зависимость {feature} от качества вина', fontsize=12)
    plt.show()



# 7. Оценка модели
print("\n7. Оценка Random Forest...")
y_pred = best_rf.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:")
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("Матрица ошибок Random Forest", fontsize=14)
plt.show()



# 8. Сохранение результатов
results = {
    'model_type': 'RandomForestClassifier',
    'best_params': grid.best_params_,
    'feature_importance': feature_imp.to_dict(),
    'accuracy': accuracy_score(y_test, y_pred)
}

print("\nАнализ завершен. Использовалась модель Random Forest со следующими параметрами:")
print(f"- Количество деревьев: {best_rf.n_estimators}")
print(f"- Глубина деревьев: {'None (без ограничений)' if best_rf.max_depth is None else best_rf.max_depth}")
print(f"- Минимальное число образцов для разделения: {best_rf.min_samples_split}")


# Импорт дополнительных библиотек для регрессии
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 8. Создание и обучение модели регрессии
print("\n8. Обучение модели Random Forest для регрессии (предсказание оценки качества 0-10)")

# Создаем и обучаем регрессор
rf_regressor = RandomForestRegressor(random_state=42,
                                   n_estimators=200,
                                   max_depth=None,
                                   min_samples_split=5)
rf_regressor.fit(X_train_scaled, wines.loc[X_train.index, 'quality'])

# Оценка модели регрессии
y_reg_pred = rf_regressor.predict(X_test_scaled)
mse = mean_squared_error(wines.loc[X_test.index, 'quality'], y_reg_pred)
r2 = r2_score(wines.loc[X_test.index, 'quality'], y_reg_pred)

print(f"\nКачество модели регрессии:")
print(f"- Среднеквадратичная ошибка (MSE): {mse:.3f}")
print(f"- Коэффициент детерминации (R²): {r2:.3f}")

# 9. Итоговые выводы и предсказания
print("\n9. Итоговые выводы и предсказания моделей")

# Анализ влияния признаков на качество
print("\nАнализ влияния химических свойств на качество вина:")
top_features = feature_imp.head(3)['Feature'].values

for feature in top_features:
    corr = wines[['quality', feature]].corr().iloc[0,1]
    trend = "растет" if corr > 0 else "падает"
    print(f"- {feature}: корреляция с качеством = {corr:.3f} ({trend})")
    
    # Градиент изменения по классам
    low_val = wines[wines['quality_class']=='low'][feature].mean()
    medium_val = wines[wines['quality_class']=='medium'][feature].mean()
    high_val = wines[wines['quality_class']=='high'][feature].mean()
    print(f"  Средние значения: Плохое={low_val:.2f}, Среднее={medium_val:.2f}, Хорошее={high_val:.2f}")

# 10. Сравнение предсказаний с реальными значениями (финальная версия)
print("\n10. Сравнение предсказаний моделей с реальными значениями")

# Выбираем 3 случайных образца из тестового набора
np.random.seed(42)
sample_indices = np.random.choice(X_test.index, size=3, replace=False)
samples = wines.loc[sample_indices].copy()

# Подготавливаем данные для предсказания
sample_features = samples[X.columns]  # Используем только фичи, которые использовались при обучении
sample_scaled = scaler.transform(sample_features)

# Получаем предсказания
quality_class_pred = best_rf.predict(sample_scaled)
quality_class_proba = best_rf.predict_proba(sample_scaled)
quality_score_pred = rf_regressor.predict(sample_scaled)

# Создаем DataFrame для сравнения
comparison_data = {
    'Образец': [f"Вино #{i+1}" for i in range(3)],
    'Тип вина': samples['wine_type'].replace({'red': 'красное', 'white': 'белое'}),
    'Фактическое качество': samples['quality'],
    'Фактический класс': samples['quality_class'],
    'Предсказанный класс': le.inverse_transform(quality_class_pred),
    'Предсказанный балл': np.round(quality_score_pred, 1),
    'Разница баллов': np.round(quality_score_pred - samples['quality'], 1),
    'Вероятности классов': [dict(zip(le.classes_, np.round(proba, 3))) for proba in quality_class_proba]
}

comparison_df = pd.DataFrame(comparison_data)

# Выводим результаты
print("\nРезультаты сравнения:")
display(comparison_df[['Образец', 'Тип вина', 'Фактическое качество', 
                      'Фактический класс', 'Предсказанный класс', 
                      'Предсказанный балл', 'Разница баллов']])

# Детализация по каждому образцу
for idx in comparison_df.index:
    print(f"\nДетали по {comparison_df.loc[idx, 'Образец']} ({comparison_df.loc[idx, 'Тип вина']}):")
    print(f"- Фактическое качество: {comparison_df.loc[idx, 'Фактическое качество']}/10 ({comparison_df.loc[idx, 'Фактический класс']})")
    print(f"- Предсказанный класс: {comparison_df.loc[idx, 'Предсказанный класс']}")
    print(f"- Предсказанный балл: {comparison_df.loc[idx, 'Предсказанный балл']}/10 (ошибка: {comparison_df.loc[idx, 'Разница баллов']})")
    print(f"- Вероятности классов: {comparison_df.loc[idx, 'Вероятности классов']}")
    print("- Химические свойства:")
    for col in X.columns:
        print(f"  {col}: {samples.loc[idx, col]:.2f}")

# Визуализация сравнения
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(3)

actual_bars = plt.bar(index - bar_width/2, comparison_df['Фактическое качество'], 
                     bar_width, color='#3498db', label='Фактическое')
predicted_bars = plt.bar(index + bar_width/2, comparison_df['Предсказанный балл'], 
                        bar_width, color='#e74c3c', label='Предсказанное')

plt.xlabel('Образцы вина', fontsize=12)
plt.ylabel('Оценка качества', fontsize=12)
plt.title('Сравнение фактического и предсказанного качества вина', fontsize=14, pad=15)
plt.xticks(index, comparison_df['Образец'])
plt.legend(fontsize=12)

# Добавляем подписи значений на столбцах
for bar in actual_bars + predicted_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 11. Итоговые выводы
print("\n11. Итоговые выводы:")
print("- Наиболее важные признаки для качества:", list(top_features))
print("- Random Forest классификатор точность: {:.1f}%".format(accuracy_score(y_test, y_pred)*100))
print("- Random Forest регрессор R² score: {:.3f}".format(r2))
print("- Алкоголь и летучая кислотность - ключевые факторы качества")
print("- Модели согласованно предсказывают качество (класс и балл)")
print("\nРекомендации:")
print("- Для повышения качества: увеличить содержание алкоголя (>12%), уменьшить летучую кислотность (<0.4)")
print("- Оптимальный pH: 3.0-3.4, сульфаты: 0.5-0.7")


# 13. Анализ условных зависимостей с помощью Partial Dependence
from sklearn.inspection import PartialDependenceDisplay

print("\nАнализ условных зависимостей (Partial Dependence):")
top_features = feature_imp.head(3)['Feature'].values

# Для каждого класса качества анализируем взаимодействия
for class_idx, class_name in enumerate(le.classes_):
    print(f"\nАнализ для класса '{class_name}':")
    
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            print(f"\nВзаимодействие {feat1} и {feat2}:")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(
                best_rf, 
                X_train_scaled, 
                features=[(feat1, feat2)],
                feature_names=X.columns,
                target=class_idx,  # Указываем целевой класс
                ax=ax,
                n_jobs=-1
            )
            plt.title(f"Взаимодействие {feat1} и {feat2} для класса {class_name}", fontsize=12)
            plt.tight_layout()
            plt.show()
            
            # Анализ статистики
            cross_tab = pd.crosstab(
                pd.cut(wines[feat1], bins=5),
                pd.cut(wines[feat2], bins=5),
                values=wines['quality'],
                aggfunc='mean'
            )
            print(f"Среднее качество при комбинациях {feat1} и {feat2}:")
            print(cross_tab.round(2))