1. Исходный код (5 функций и сортировки)
Файл: src/my_functions.py
Здесь 5 функций разного типа для демонстрации различных тестовых сценариев.
import math
import logging
from typing import List, Optional

def calculate_circle_area(radius: float) -> float:
    """
    Вычисляет площадь круга по радиусу.
    Генерирует ValueError для отрицательного радиуса.
    """
    if radius < 0:
        raise ValueError("Радиус не может быть отрицательным")
    return math.pi * (radius ** 2)

def is_palindrome(text: str) -> bool:
    """
    Проверяет, является ли строка палиндромом (игнорирует регистр и пробелы).
    """
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum())
    return cleaned_text == cleaned_text[::-1]

def merge_and_sort_unique_lists(list1: List[int], list2: List[int]) -> List[int]:
    """
    Объединяет два списка целых чисел, оставляет только уникальные значения
    и возвращает отсортированный по возрастанию список.
    """
    merged_set = set(list1) | set(list2)
    return sorted(merged_set)

def get_statistics(numbers: List[float]) -> Optional[dict]:
    """
    Вычисляет базовую статистику для списка чисел.
    Возвращает словарь с ключами: mean, median, min, max.
    Для пустого списка возвращает None.
    """
    if not numbers:
        return None

    sorted_numbers = sorted(numbers)
    n = len(numbers)

    stats = {
        'mean': sum(numbers) / n,
        'median': (sorted_numbers[n // 2] if n % 2 != 0
                   else (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2),
        'min': min(numbers),
        'max': max(numbers)
    }
    return stats

class BankAccount:
    """Простой класс банковского счёта для демонстрации тестирования состояний."""

    def __init__(self, owner: str, initial_balance: float = 0.0):
        self.owner = owner
        self._balance = initial_balance
        self.transaction_log = []

    def deposit(self, amount: float):
        if amount <= 0:
            raise ValueError("Сумма депозита должна быть положительной")
        self._balance += amount
        self.transaction_log.append(f"Депозит: +{amount}")

    def withdraw(self, amount: float):
        if amount <= 0:
            raise ValueError("Сумма снятия должна быть положительной")
        if amount > self._balance:
            raise ValueError("Недостаточно средств на счёте")
        self._balance -= amount
        self.transaction_log.append(f"Снятие: -{amount}")

    @property
    def balance(self):
        return self._balance
Файл: src/sorting_algorithms.py
Адаптированные функции сортировки из ЛР3 без логирования внутри, чтобы их было проще тестировать.
from typing import List, Tuple

def bubble_sort(arr: List[int]) -> Tuple[List[int], int]:
    """Пузырьковая сортировка. Возвращает отсортированный список и количество операций."""
    n = len(arr)
    operations = 0
    arr = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            operations += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                operations += 1  # Учитываем обмен
    return arr, operations

def quick_sort(arr: List[int]) -> Tuple[List[int], int]:
    """Быстрая сортировка. Возвращает отсортированный список и количество операций."""
    def _quick_sort(inner_arr: List[int], ops: List[int]) -> List[int]:
        if len(inner_arr) <= 1:
            return inner_arr
        pivot = inner_arr[len(inner_arr) // 2]
        left = [x for x in inner_arr if x < pivot]
        middle = [x for x in inner_arr if x == pivot]
        right = [x for x in inner_arr if x > pivot]
        ops[0] += len(inner_arr)  # Учитываем сравнения при создании подсписков
        return _quick_sort(left, ops) + middle + _quick_sort(right, ops)

    operations = [0]
    sorted_arr = _quick_sort(arr, operations)
    return sorted_arr, operations[0]

# ... здесь можно добавить selection_sort, insertion_sort и другие
2. Тестирование с использованием возможностей pytest
Файл: tests/test_my_functions.py
Здесь используются ключевые возможности pytest, описанные в статье на Хабре.
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.my_functions import (
    calculate_circle_area,
    is_palindrome,
    merge_and_sort_unique_lists,
    get_statistics,
    BankAccount
)

# Тест 1: Параметризация — мощный инструмент для тестирования множества входных данных
@pytest.mark.parametrize("radius, expected_area", [
    (1.0, 3.141592653589793),
    (0.0, 0.0),
    (5.0, 78.53981633974483),
])
def test_calculate_circle_area_valid(radius, expected_area):
    """Тестируем функцию с корректными радиусами."""
    assert calculate_circle_area(radius) == pytest.approx(expected_area)

# Тест 2: Проверка выброса исключений
def test_calculate_circle_area_negative_radius():
    """Ожидаем ValueError при отрицательном радиусе."""
    with pytest.raises(ValueError, match="Радиус не может быть отрицательным"):
        calculate_circle_area(-5)

# Тест 3: Ещё параметризация для строковых функций
@pytest.mark.parametrize("text, expected", [
    ("А роза упала на лапу Азора", True),
    ("Python", False),
    ("12321", True),
    ("", True),  # Пустая строка — палиндром
    ("a", True),
])
def test_is_palindrome(text, expected):
    assert is_palindrome(text) == expected

# Тест 4: Фикстуры — для подготовки и переиспользования данных
@pytest.fixture
def sample_lists():
    """Фикстура, возвращающая тестовые списки."""
    return [3, 1, 2], [4, 2, 5]

def test_merge_and_sort_unique_lists(sample_lists):
    """Используем фикстуру sample_lists."""
    list1, list2 = sample_lists
    result = merge_and_sort_unique_lists(list1, list2)
    assert result == [1, 2, 3, 4, 5]
    # Проверяем, что результат действительно отсортирован
    assert result == sorted(result)
    # Проверяем, что в результате нет дубликатов
    assert len(result) == len(set(result))

# Тест 5: Работа с возвращаемыми словарями и None
def test_get_statistics_normal():
    numbers = [10.0, 20.0, 30.0]
    stats = get_statistics(numbers)
    assert stats is not None
    assert stats['mean'] == 20.0
    assert stats['median'] == 20.0
    assert stats['min'] == 10.0
    assert stats['max'] == 30.0

def test_get_statistics_empty():
    """Для пустого списка функция должна вернуть None."""
    assert get_statistics([]) is None

# Тест 6: Группировка тестов с помощью класса (модульность)
class TestBankAccount:
    """Группа тестов для класса BankAccount."""

    @pytest.fixture
    def account(self):
        """Фикстура создаёт новый экземпляр счёта для каждого теста в классе."""
        return BankAccount("Иван Иванов", 100.0)

    def test_initial_balance(self, account):
        """Тест начального баланса."""
        assert account.balance == 100.0
        assert account.owner == "Иван Иванов"

    def test_deposit(self, account):
        """Тест внесения депозита."""
        account.deposit(50.0)
        assert account.balance == 150.0
        assert len(account.transaction_log) == 1

    def test_withdraw_sufficient_funds(self, account):
        """Тест снятия средств при достаточном балансе."""
        account.withdraw(30.0)
        assert account.balance == 70.0

    @pytest.mark.parametrize("invalid_amount", [-10.0, 0.0])
    def test_deposit_invalid_amount_raises_error(self, account, invalid_amount):
        """Параметризованный тест на неверную сумму депозита."""
        with pytest.raises(ValueError):
            account.deposit(invalid_amount)

    def test_withdraw_insufficient_funds_raises_error(self, account):
        """Тест на попытку снятия при недостатке средств."""
        with pytest.raises(ValueError, match="Недостаточно средств"):
            account.withdraw(200.0)
Файл: tests/test_sorting_algorithms.py
Тестируем корректность и эффективность (сравниваем количество операций).
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.sorting_algorithms import bubble_sort, quick_sort

# Тест 1: Параметризация для проверки корректности сортировки
@pytest.mark.parametrize("sort_func, input_list, expected", [
    (bubble_sort, [64, 34, 25, 12, 22, 11, 90], [11, 12, 22, 25, 34, 64, 90]),
    (bubble_sort, [], []),
    (bubble_sort, [5], [5]),
    (quick_sort, [3, 6, 8, 10, 1, 2, 1], [1, 1, 2, 3, 6, 8, 10]),
    (quick_sort, [], []),
    (quick_sort, [7], [7]),
])
def test_sorting_algorithms_correctness(sort_func, input_list, expected):
    """Тестируем, что алгоритмы корректно сортируют разные типы списков."""
    sorted_arr, _ = sort_func(input_list)
    assert sorted_arr == expected

# Тест 2: Проверка эффективности (количества операций)
def test_sorting_efficiency():
    """
    Сравниваем количество операций для разных алгоритмов на одном наборе данных.
    Быстрая сортировка должна быть эффективнее пузырьковой на больших данных.
    """
    test_data = [64, 34, 25, 12, 22, 11, 90]
    _, bubble_ops = bubble_sort(test_data)
    _, quick_ops = quick_sort(test_data)
    # На небольшом массиве разница может быть невелика, но quick_sort должен иметь меньше операций
    print(f"\nПузырьковая сортировка: {bubble_ops} операций")
    print(f"Быстрая сортировка: {quick_ops} операций")
    assert quick_ops < bubble_ops  # Это основное утверждение теста

# Тест 3: Фикстура для генерации тестовых данных
@pytest.fixture(params=[0, 1, 10, 100])
def random_list(request):
    """Фикстура генерирует списки разной длины для тестов производительности."""
    import random
    n = request.param
    return [random.randint(-1000, 1000) for _ in range(n)]

def test_sorting_stress(random_list):
    """Нагрузочный тест: проверяем, что алгоритмы не ломаются на случайных данных."""
    sorted_bubble, _ = bubble_sort(random_list)
    sorted_quick, _ = quick_sort(random_list)
    # Оба алгоритма должны дать одинаковый отсортированный результат
    assert sorted_bubble == sorted_quick
    # И этот результат должен быть действительно отсортирован
    assert sorted_bubble == sorted(sorted_bubble)