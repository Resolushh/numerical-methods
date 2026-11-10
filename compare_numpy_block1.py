import sys
import time
import math
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

rng = np.random.default_rng(42)
# Списки для циклов


def measure(fn, data):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        result = fn(data)
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
    return result, best


def compare(fn_python, fn_numpy, data_python, data_numpy):
    result_python, t_python = measure(fn_python, data_python)
    result_numpy, t_numpy = measure(fn_numpy, data_numpy)
    print(f"Python: {t_python * 1000:.3f} мс")
    print(f"NumPy: {t_numpy * 1000:.3f} мс")
    print(f"Ускорение: {t_python / t_numpy:.2f} раз")
    print("Результаты совпадают:", np.allclose(result_python, result_numpy))
    return result_python, result_numpy


# 1. Сумма
def sum_python(a):
    total = 0.0
    for x in a:
        total += x
    return total


def sum_numpy(a):
    return a.sum()


print("Задача 1")
print("n | Python, мс | NumPy, мс | Ускорение")
sizes = [100_000, 1_000_000, 10_000_000]
speeds = []
for n in sizes:
    a = rng.normal(size=n)
    values = a.tolist()
    r1, t1 = measure(sum_python, values)
    r2, t2 = measure(sum_numpy, a)
    speeds.append(t1 / t2)
    print(f"{n} | {t1 * 1000:.3f} | {t2 * 1000:.3f} | {t1 / t2:.2f}")
    print("Совпадение:", np.isclose(r1, r2))
first_n = 0
for i in range(len(sizes)):
    if speeds[i] > 10:
        first_n = sizes[i]
        break
if first_n > 0:
    print("Первый проверенный размер с ускорением больше 10 раз:", first_n)
else:
    print("На проверенных размерах ускорения больше 10 раз нет.")
print("В NumPy сумма считается одной строкой. Ускорение не растет ровно с размером массива: на больших данных влияет скорость памяти.")
del a, values


# 2. Среднее и стандартное отклонение
def stats_python(a):
    total = 0.0
    squares = 0.0
    for x in a:
        total += x
        squares += x * x
    mean = total / len(a)
    std = math.sqrt(squares / len(a) - mean * mean)
    return mean, std


def stats_numpy(a):
    return a.mean(), a.std()


print("\nЗадача 2")
a = rng.normal(size=1_000_000)
values = a.tolist()
r1, r2 = compare(stats_python, stats_numpy, values, a)
print("Совпадение до 1e-12:", np.allclose(r1, r2, atol=1e-12, rtol=0))
print("mean и std проще читать, чем цикл с суммами. std сначала находит среднее, затем считает отклонения от него, так меньше теряется точность при вычитании близких больших чисел.")


# 3. sin(x) + x^2
def transform_python(a):
    result = []
    for x in a:
        result.append(math.sin(x) + x * x)
    return result


def transform_numpy(a):
    return np.sin(a) + a ** 2


print("\nЗадача 3")
r1, r2 = compare(transform_python, transform_numpy, values, a)
print("Список с числами примерно:", len(r1) * 32 / 1e6, "МБ")
print("Данные массива:", r2.nbytes / 1e6, "МБ")
print("С NumPy получается одна строка вместо цикла. Кроме результата нужны два временных массива: для sin(a) и a**2, по 8 МБ каждый.")


# 4. Ограничение значений
def clip_python(a):
    result = []
    for x in a:
        if x < 0:
            result.append(0.0)
        elif x > 1:
            result.append(1.0)
        else:
            result.append(x)
    return result


def clip_numpy(a):
    return np.clip(a, 0, 1)


print("\nЗадача 4")
compare(clip_python, clip_numpy, values, a)
print("Изменено элементов:", np.sum((a < 0) | (a > 1)))
print("Без строки def получилось 9 строк у цикла и 1 у NumPy, то есть в 9 раз короче. В clip сразу видны границы 0 и 1, без for и if/elif.")


# 5. Суммы строк
def rows_python(m):
    result = []
    for row in m:
        total = 0.0
        for x in row:
            total += x
        result.append(total)
    return result


def rows_numpy(m):
    return m.sum(axis=1)


print("\nЗадача 5")
matrix = rng.normal(size=(100_000, 10))
rows = matrix.tolist()
compare(rows_python, rows_numpy, rows, matrix)
print("Без def получилось 7 строк против 1. sum(axis=1) проще поддерживать: сразу видно, что складываются строки, не нужно разбирать два цикла.")


# 6. Вычитание средних столбцов
def center_python(m):
    means = [0.0] * len(m[0])
    for j in range(len(m[0])):
        for i in range(len(m)):
            means[j] += m[i][j]
        means[j] /= len(m)
    result = []
    for i in range(len(m)):
        row = []
        for j in range(len(m[0])):
            row.append(m[i][j] - means[j])
        result.append(row)
    return result


def center_numpy(m):
    return m - m.mean(axis=0)


print("\nЗадача 6")
r1, r2 = compare(center_python, center_numpy, rows, matrix)
print("Средние около нуля:", np.allclose(r2.mean(axis=0), 0, atol=1e-12))
print("Без def получилось 12 строк против 1. В цикле надо следить за i и j, а NumPy сам вычитает среднее каждого столбца.")


# 7. Максимум и индекс
def maximum_python(a):
    maximum = a[0]
    index = 0
    for i in range(1, len(a)):
        if a[i] > maximum:
            maximum = a[i]
            index = i
    return maximum, index


def maximum_numpy(a):
    return a.max(), a.argmax()


print("\nЗадача 7")
r1, r2 = compare(maximum_python, maximum_numpy, values, a)
print("Индексы совпадают точно:", r1[1] == r2[1])
with_nan = a.copy()
with_nan[0] = np.nan
print("Цикл с NaN в начале:", maximum_python(with_nan.tolist()))
print("nanmax и nanargmax:", np.nanmax(with_nan), np.nanargmax(with_nan))
print("max и argmax проще ручного поиска. Если первым стоит NaN, цикл оставляет его максимумом, а nanmax ищет среди остальных чисел.")


# 8. Положительные элементы
def positive_python(a):
    return [x for x in a if x > 0]


def positive_numpy(a):
    return a[a > 0]


print("\nЗадача 8")
r1, r2 = compare(positive_python, positive_numpy, values, a)
print("Список с числами примерно:", len(r1) * 32 / 1e6, "МБ")
print("Данные массива:", r2.nbytes / 1e6, "МБ")
print("Оба варианта короткие и понятные. Массив занимает примерно в 4 раза меньше памяти: 8 байт на число вместо 24 байт на число и 8 на ссылку.")
print("Память списка указана вместе с числами, хотя при отборе копируются только ссылки на них.")


# 9. Модуль на месте
def abs_python(a):
    for i in range(len(a)):
        if a[i] < 0:
            a[i] = -a[i]
    return a


def abs_numpy(a):
    np.abs(a, out=a)
    return a


print("\nЗадача 9")
t1 = float("inf")
t2 = float("inf")
for _ in range(3):
    test_python = values.copy()
    test_numpy = a.copy()
    start = time.perf_counter()
    r1 = abs_python(test_python)
    t1 = min(t1, time.perf_counter() - start)
    start = time.perf_counter()
    r2 = abs_numpy(test_numpy)
    t2 = min(t2, time.perf_counter() - start)
print(f"Python: {t1 * 1000:.3f} мс")
print(f"NumPy: {t2 * 1000:.3f} мс")
print(f"Ускорение: {t1 / t2:.2f} раз")
print("Результаты совпадают:", np.allclose(r1, r2))
print("Цикл вернул тот же список:", r1 is test_python)
print("NumPy вернул тот же массив:", r2 is test_numpy)
print("out=a записывает ответ в тот же массив, это проще цикла и не требует места под второй массив. Копии перед повторами нужны для одинаковых исходных данных, их создание не входит в замер.")


# 10. Два условия
def interval_python(a):
    result = []
    for x in a:
        if -1 < x < 1:
            result.append(x)
    return result


def interval_numpy(a):
    return a[(a > -1) & (a < 1)]


print("\nЗадача 10")
compare(interval_python, interval_numpy, values, a)
print("В Python условие читается проще, зато в NumPy весь отбор занимает одну строку. Скобки обязательны, потому что & выполняется раньше сравнений, а and для двух массивов не подходит.")


# 11. Расстояния
def distances_python(data):
    x, y = data
    result = []
    for i in range(len(x)):
        row = []
        for j in range(len(y)):
            total = 0.0
            for k in range(3):
                total += (x[i][k] - y[j][k]) ** 2
            row.append(math.sqrt(total))
        result.append(row)
    return result


def distances_numpy(data):
    x, y = data
    difference = x[:, None, :] - y[None, :, :]
    return np.sqrt((difference ** 2).sum(axis=2))


print("\nЗадача 11")
x = rng.normal(size=(1000, 3))
y = rng.normal(size=(1000, 3))
compare(distances_python, distances_numpy, (x.tolist(), y.tolist()), (x, y))
print("Массив разностей: 1000 * 1000 * 3 * 8 = 24 МБ, плюс память на квадраты и результат.")
print("NumPy считает быстрее и короче, но временные массивы занимают много памяти. Можно считать расстояния по частям или через суммы квадратов и X @ Y.T, чтобы не создавать массив (1000, 1000, 3).")
