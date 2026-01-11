import csv
from itertools import combinations

def simple_apriori(transactions, min_support=0.5):
    """
    Реализация алгоритма Apriori

    transactions: список транзакций (каждая транзакция - список товаров)
    min_support: минимальная поддержка (например, 0.5 = 50%)
    """

    # Шаг 1: Считаем поддержку для отдельных товаров
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1

    # Преобразуем в поддержку (доля транзакций)
    num_transactions = len(transactions)
    frequent_items = {}

    # Находим частые одиночные товары (отсортированные)
    frequent_items[1] = {}
    for item, count in sorted(item_counts.items()):
        support = count / num_transactions
        if support >= min_support:
            frequent_items[1][(item,)] = support  # Используем кортежи

    k = 1
    # Пока находим частые наборы
    while frequent_items.get(k):
        print(f"\nЧастые наборы из {k} товаров:")
        for itemset, support in frequent_items[k].items():
            print(f"  {set(itemset)} - поддержка: {support:.2%}")

        # Создаем кандидатов для следующего размера
        candidates = generate_candidates(list(frequent_items[k].keys()), k)

        # Проверяем поддержку кандидатов
        next_frequents = {}
        for candidate in candidates:
            support = calculate_support(candidate, transactions)
            if support >= min_support:
                next_frequents[candidate] = support

        if not next_frequents:
            break

        frequent_items[k + 1] = next_frequents
        k += 1

    return frequent_items


def generate_candidates(frequent_k, k):
    """Генерирует кандидатов размера k+1 из частых наборов размера k"""
    candidates = set()
    n = len(frequent_k)
    frequent_k_set = set(frequent_k) if k > 1 else None
    
    for i in range(n):
        for j in range(i + 1, n):
            # Объединяем два набора
            itemset1 = frequent_k[i]
            itemset2 = frequent_k[j]
            
            if k == 1:
                # Для одиночных элементов просто объединяем
                new_candidate = tuple(sorted([itemset1[0], itemset2[0]]))
                candidates.add(new_candidate)
            else:
                # Для k > 1: проверяем, что первые k-1 элементов совпадают
                if itemset1[:-1] == itemset2[:-1]:
                    # Объединяем последние элементы
                    new_candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                    
                    # Проверяем, что все (k-1)-подмножества частые (apriori property)
                    if has_infrequent_subset(new_candidate, frequent_k_set, k):
                        continue
                    
                    candidates.add(new_candidate)
    
    return sorted(candidates)


def has_infrequent_subset(candidate, frequent_k_set, subset_size):
    """Проверяет, есть ли у кандидата нечастые подмножества размера subset_size"""
    for subset in combinations(candidate, subset_size):
        subset = tuple(sorted(subset))
        if subset not in frequent_k_set:
            return True
    return False


def calculate_support(itemset, transactions):
    """Считает поддержку для набора товаров"""
    count = 0
    itemset_set = set(itemset)
    for transaction in transactions:
        if itemset_set.issubset(set(transaction)):
            count += 1
    return count / len(transactions)


def generate_rules(frequent_items, transactions, min_confidence=0.7):
    """
    Генерирует правила ассоциации по частым наборам.

    frequent_items: словарь вида
        k -> { itemset (кортеж товаров) : support (поддержка) }
    min_confidence: минимально допустимая достоверность правила
    """
    rules = []

    # Подготовим быстрый доступ к поддержкам для любых наборов
    # (чтобы не пересчитывать поддержку каждый раз с нуля)
    support_lookup = {}
    for size_dict in frequent_items.values():
        support_lookup.update(size_dict)

    for k in frequent_items:
        if k < 2:  # Для правил нужны наборы из 2+ товаров
            continue

        for itemset, support_full in frequent_items[k].items():
            itemset_set = set(itemset)

            # Генерируем все возможные правила
            for i in range(1, k):
                # Все комбинации для антецедента (левой части правила)
                for antecedent in combinations(itemset, i):
                    antecedent_set = set(antecedent)
                    consequent_set = itemset_set - antecedent_set

                    # Берём поддержку антецедента (если нет в словаре — считаем)
                    antecedent_tuple = tuple(sorted(antecedent_set))
                    support_antecedent = support_lookup.get(
                        antecedent_tuple,
                        calculate_support(antecedent_tuple, transactions)
                    )
                    confidence = support_full / support_antecedent if support_antecedent > 0 else 0

                    if confidence >= min_confidence:
                        # Берём поддержку консеквента (правой части правила)
                        consequent_tuple = tuple(sorted(consequent_set))
                        support_consequent = support_lookup.get(
                            consequent_tuple,
                            calculate_support(consequent_tuple, transactions)
                        )

                        # Lift = доверие(A -> B) / support(B)
                        lift = confidence / support_consequent if support_consequent > 0 else 0

                        rules.append({
                            'antecedent': antecedent_set,
                            'consequent': consequent_set,
                            'support': support_full,
                            'confidence': confidence,
                            'lift': lift
                        })

    return rules


def read_transactions_from_csv(file_path):
    """
    Читает транзакции из CSV файла.
    Предполагается, что каждая строка - это транзакция, товары разделены запятыми.
    Без заголовков.
    """
    transactions = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Удаляем пустые строки и очищаем товары от лишних пробелов
            cleaned_row = [item.strip() for item in row if item.strip()]
            if cleaned_row:
                transactions.append(cleaned_row)
    return transactions


# Тестовый пример
def test_simple_apriori(csv_file_path):
    # Читаем транзакции из CSV
    transactions = read_transactions_from_csv(csv_file_path)

    print("Транзакции:")
    for i, t in enumerate(transactions, 1):
        print(f"{i}: {t}")

    print("\n" + "=" * 50)
    print("ЗАПУСК APRIORI")
    print("=" * 50)

    # Запускаем Apriori
    min_support = 0.4  # 40%
    frequent_items = simple_apriori(transactions, min_support)

    print("\n" + "=" * 50)
    print("ГЕНЕРАЦИЯ ПРАВИЛ")
    print("=" * 50)

    # Генерируем правила
    min_confidence = 0.6 # 60%
    rules = generate_rules(frequent_items, transactions, min_confidence)

    print(f"\nНайдено правил: {len(rules)}")
    for i, rule in enumerate(rules, 1):
        print(f"\nПравило {i}:")
        print(f"  {rule['antecedent']} → {rule['consequent']}")
        print(f"  Поддержка: {rule['support']:.2%}")
        print(f"  Достоверность: {rule['confidence']:.2%}")
        print(f"  Lift: {rule['lift']:.2f}")


# Пример использования
if __name__ == "__main__":
    test_simple_apriori(r'C:\Users\Пользователь\Desktop\програмироание\mecanicoratrismeckanikkavenal\lab-4\apriori_db_1.csv')


# Поддержка (support) – доля всех транзакций, где одновременно есть и левая, и правая часть правила.
# Достоверность (confidence) – среди транзакций, где есть левая часть (A), в какой доле также есть правая (B).
# Lift – во сколько раз правило A -> B сильнее (или слабее) простой совместной встречаемости B.