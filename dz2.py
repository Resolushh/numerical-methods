import math
import random
from decimal import Decimal, localcontext
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAIRS = [
    ("3.14 и pi", 3.14, math.pi),
    ("2.718 и e", 2.718, math.e),
    ("9.8 и 9.81", 9.8, 9.81),
    ("0.333 и 1/3", 0.333, 1 / 3),
]


def relative_error(approx, exact):
    if exact == 0:
        return float("inf")
    return abs(approx - exact) / abs(exact)


def naive_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def table(headers, rows):
    print(" | ".join(headers))
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.10g}")
            else:
                cells.append(str(value))
        print(" | ".join(cells))


def error_row(name, value, reference):
    return name, repr(value), abs(value - reference), relative_error(value, reference)


def task_1():
    print("Delta = |x - x*|, delta в процентах = Delta / |x*| * 100.")
    rows = [(name, abs(x - exact), 100 * relative_error(x, exact))
            for name, x, exact in PAIRS]
    table(("Пара", "Delta", "delta, %"), rows)
    by_absolute = sorted(rows, key=lambda row: row[1])
    by_relative = sorted(rows, key=lambda row: row[2])
    print("По Delta от лучшего: " + ", ".join(row[0] for row in by_absolute))
    print("По delta от лучшего: " + ", ".join(row[0] for row in by_relative))
    print("Порядок меняется из-за деления на точное значение.")
    print("У 0.333 меньше абсолютная ошибка, чем у 3.14, но больше относительная.")


def task_2():
    table(("Измерение", "Delta", "delta, %"), [
        ("Мост", f"{abs(1247 - 1250)} м", 100 * relative_error(1247, 1250)),
        ("Болт", f"{abs(9.7 - 10.0):.1f} мм", 100 * relative_error(9.7, 10.0)),
    ])
    print("По абсолютной ошибке больше промах с мостом: 3 м против 0.0003 м.")
    print("По доле от размера сильнее ошиблись с болтом: 3% против 0.24%.")


def task_3():
    table(("Пара", "relative_error"),
          [(name, relative_error(x, exact)) for name, x, exact in PAIRS])
    print(f"relative_error(1, 0) = {relative_error(1, 0)}")
    print("Для процентов умножаю результат на 100.")
    print("При точном нуле возвращаю inf по условию.")


def task_4():
    x, delta = 47.3, 0.001
    lower, upper = x / (1 + delta), x / (1 - delta)
    print("Из |x/x* - 1| <= 0.001 получаю x/1.001 <= x* <= x/0.999.")
    print(f"Точные границы: [47.3/1.001, 47.3/0.999], приближенно [{lower:.5f}, {upper:.5f}].")
    print("Интервал несимметричен из-за x* в знаменателе.")


def task_5():
    threshold = 0.3 / 0.005
    print(f"|T| * 0.005 <= 0.3, значит |T| <= {threshold:g} °C.")
    print("При |T| < 60 °C точнее B, при |T| = 60 °C ошибки равны по верхней границе.")
    print("При |T| > 60 °C точнее A.")


def task_6():
    table(("Число", "Значащих цифр", "Экспоненциальная запись"), [
        ("0.004520", "4", "4.520e-3"),
        ("1200", "2, 3 или 4", "1.2e3 / 1.20e3 / 1.200e3"),
        ("1.200", "4", "1.200e0"),
        ("10.03", "4", "1.003e1"),
        ("0.03050", "4", "3.050e-2"),
        ("250000", "от 2 до 6", "2.5e5 / 2.50e5 / 2.500e5 / 2.5000e5 / 2.50000e5"),
    ])
    print("В 1200 и 250000 непонятно, какие нули значащие.")
    print("В экспоненциальной записи значащие нули видны в мантиссе.")


def task_7():
    print(f"{4.738291:.2f} ± 0.02, осталось 3 значащие цифры.")
    print("Погрешность дана в сотых, поэтому результат округляю до сотых.")
    print("Цифры после сотых создают видимость лишней точности.")


def task_8():
    halves = (0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5)
    print("round:", *(round(value) for value in halves))
    rng = random.Random(42)
    values = [rng.randrange(10) + 0.5 for _ in range(10**6)]
    exact = math.fsum(values)
    banker = sum(round(value) for value in values)
    upwards = sum(math.floor(value + 0.5) for value in values)
    table(("Способ", "Сумма", "Сдвиг от исходной"), [
        ("Исходная", exact, 0), ("До четного", banker, banker - exact),
        ("Вверх", upwards, upwards - exact),
    ])
    print(f"Вверх минус до четного: {upwards - banker}.")
    print("На половинах round берет четное, и ошибки почти гасят друг друга.")
    print("Округление вверх добавляет по 0.5 и ведет к переплате.")


def task_9():
    values = [1 / k for k in range(1, 101)]
    reference = math.fsum(values)
    plain = naive_sum(values)
    rounded = naive_sum(round(value, 3) for value in values)
    table(("Вариант", "S", "Delta S", "delta S"), [
        error_row("Цикл", plain, reference),
        error_row("С округлением", rounded, reference),
        error_row("Эталон fsum", reference, reference),
    ])
    if plain == reference:
        print("Цикл совпал с эталоном, поэтому конечного отношения ошибок нет.")
    else:
        ratio = abs(rounded - reference) / abs(plain - reference)
        print(f"Отношение ошибок округленного варианта и обычного цикла: {ratio:.6g}.")
    print("Округление каждого слагаемого портит сумму еще до вывода.")
    print("Округляю только итог.")


def task_10():
    value, reference = 0.1 + 0.2, 0.3
    print(f"{value!r}, ==: {value == reference}, isclose: {math.isclose(value, reference)}")
    print(f"Delta = {abs(value - reference):.8g}, delta = {relative_error(value, reference):.8g}.")
    print("Десятичная дробь 0.1 в двоичной записи бесконечна.")
    print("В float число приходится округлять.")
    print("Сумма округленных чисел чуть отличается от 0.3.")


def task_11():
    a, b = 1 + 1e-15, 1.0
    value = a - b
    print(f"a - b = {value:.17g}, ожидалось 1e-15, delta = {relative_error(value, 1e-15):.6%}.")
    a, b, c = 1, 10**8, 1
    disc = b*b - 4*a*c
    root_direct = (-b + math.sqrt(disc)) / (2*a)
    root_large = (-b - math.sqrt(disc)) / (2*a)
    root_stable = c / (a * root_large)
    with localcontext() as ctx:
        ctx.prec = 50
        reference = (-Decimal(b) + Decimal(disc).sqrt()) / (2*a)
        direct_error = float(relative_error(Decimal.from_float(root_direct), reference))
        stable_error = float(relative_error(Decimal.from_float(root_stable), reference))
    print("Сначала нахожу большой корень по модулю, затем малый: x2 = c/(a*x1).")
    print(f"Эталон Decimal: {reference}")
    table(("Метод", "Малый по модулю корень", "Отн. ошибка"), [
        ("Прямой", root_direct, direct_error), ("Устойчивый", root_stable, stable_error),
    ])
    print("В a - b часть цифр теряется при округлении a.")
    print(f"У устойчивой формулы ошибка меньше в {direct_error / stable_error:.2g} раз.")


def task_12():
    values = [0.1] * 1_000_000
    reference = 1e5
    loop_row = error_row("Цикл", naive_sum(values), reference)
    sum_row = error_row("sum", sum(values), reference)
    fsum_row = error_row("fsum", math.fsum(values), reference)
    rows = [loop_row, sum_row, fsum_row]
    table(("Метод", "S", "Delta", "delta"), rows)
    for name, _, error, _ in rows[1:]:
        if error:
            print(f"У {name} ошибка в {rows[0][2] / error:.6g} раз меньше, чем у цикла.")
        else:
            print(f"У {name} ошибка 0 в float, конечного отношения к ошибке цикла нет.")
    print("При сложении в цикле округление повторяется на каждом шаге.")
    print("Точность sum зависит от версии Python, fsum учитывает остатки округления.")


def task_13():
    small = [1e-10] * 500_000
    large = [1e6] * 500_000
    ascending = small + large
    descending = large + small
    reference = math.fsum(ascending)
    table(("Порядок", "S", "Delta", "delta"), [
        error_row("Маленькие, большие", naive_sum(ascending), reference),
        error_row("Большие, маленькие", naive_sum(descending), reference),
        error_row("Эталон fsum", reference, reference),
    ])
    print("При сложении с большой суммой добавка 1e-10 теряется.")
    print("Если сложить маленькие числа первыми, их вклад сохраняется.")


def task_14():
    x = 1.0
    log_factor = math.log(1.0001)
    errors = [0.0]
    for step in range(1, 10_001):
        x *= 1.0001
        reference = math.exp(step * log_factor)
        errors.append(relative_error(x, reference))
    crossings = []
    for threshold in (1e-9, 1e-6, 1e-3):
        first_step = "нет"
        for step, error in enumerate(errors):
            if error > threshold:
                first_step = step
                break
        crossings.append((threshold, first_step))
    table(("Порог", "Первый шаг выше порога"), crossings)
    print(f"x = {x:.17g}, эталон = {reference:.17g}, delta = {errors[-1]:.6g}.")
    print(f"Максимум относительной ошибки за 10000 шагов: {max(errors):.6g}.")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(range(len(errors)), errors, linewidth=1)
    ax.set(xlabel="Шаг", ylabel="Относительная ошибка", title="ДЗ 2, задача 14. Ошибка умножения")
    ax.grid(alpha=0.3)
    fig.savefig(Path(__file__).with_name("dz2_growth_error.png"), dpi=140)
    plt.close(fig)
    print("Ошибка колеблется и остается ниже всех трех порогов.")
    print("За 10000 умножений ошибка выросла примерно до 1e-14.")


def task_15():
    float_total = 0.0
    cents = 0
    with localcontext() as ctx:
        ctx.prec = 28
        cent = Decimal("0.01")
        decimal_total = Decimal("0.00")
        for _ in range(10**6):
            float_total += 0.01
            decimal_total += cent
            cents += 1
        cents_total = Decimal(cents) / 100
        difference = abs(Decimal.from_float(float_total) - decimal_total)
        percent = 100 * difference / decimal_total
    print(f"float: {float_total!r}, Decimal: {decimal_total}, int: {cents} коп. = {cents_total} руб.")
    print(f"Расхождение float с Decimal: {difference:.9g} руб., {percent:.9g}%.")
    print(f"Расхождение целых копеек с Decimal: {abs(cents_total - decimal_total):.2f} руб.")
    print("Итог 10000.00 руб. точно дают Decimal и целые копейки.")
    print("В float округление 0.01 дает ошибку при накоплении.")


def task_16():
    rng = random.Random(42)
    values = [rng.random() for _ in range(10**5)]
    n = len(values)
    methods = [("Цикл", naive_sum(values) / n), ("sum", sum(values) / n),
               ("fsum", math.fsum(values) / n), ("numpy.mean", float(np.mean(values)))]
    with localcontext() as ctx:
        ctx.prec = 80
        reference = sum(Decimal.from_float(value) for value in values) / n
        eps = Decimal.from_float(math.ulp(1.0))
        fsum_mean = methods[2][1]
        rows = []
        for name, value in methods:
            error = abs(Decimal.from_float(value) - reference) / eps
            distance = abs(value - fsum_mean) / math.ulp(1.0)
            rows.append((name, repr(value), float(error), distance))
    table(("Метод", "Среднее", "|Ошибка| / eps", "|Разность с fsum| / eps"), rows)
    best = min(row[2] for row in rows)
    print("На этой выборке ближе всего к эталону: " + ", ".join(row[0] for row in rows if row[2] == best) + ".")
    print("NumPy складывает блоками, поэтому итог может отличаться от fsum.")


def main():
    tasks = (
        ("Ручной счет", task_1), ("Мост и болт", task_2), ("Относительная ошибка", task_3),
        ("Интервал", task_4), ("Датчики", task_5), ("Значащие цифры", task_6),
        ("Точность записи", task_7), ("Округление до четного", task_8),
        ("Промежуточное округление", task_9), ("0.1 + 0.2", task_10),
        ("Потеря точности", task_11), ("Накопление в цикле", task_12),
        ("Порядок сложения", task_13), ("Ошибка умножения", task_14),
        ("Рубли и копейки", task_15), ("Бонус. Сравнение средних", task_16),
    )
    for number, (title, solve) in enumerate(tasks, 1):
        print(f"\nЗадача {number}. {title}")
        solve()


if __name__ == "__main__":
    main()
