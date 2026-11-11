import math
import platform
import random
from pathlib import Path

import matplotlib
import mpmath as mp
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def table(headers, rows):
    print(" | ".join(headers))
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.8g}")
            else:
                cells.append(str(value))
        print(" | ".join(cells))


def relative_error(value, reference):
    return float(abs(mp.mpf(float(value)) - reference) / abs(reference))


def compare(inputs, direct, stable, reference):
    rows = []
    for text in inputs:
        x = float(text)
        expected = reference(mp.mpf(text))
        first = direct(x)
        second = stable(x)
        direct_error = relative_error(first, expected)
        stable_error = relative_error(second, expected)
        rows.append((x, first, second, mp.nstr(expected, 18), direct_error, stable_error))
    return rows


def error_table(rows):
    table(("x", "Ошибка прямого", "Ошибка устойчивого"),
          [(row[0], row[4], row[5]) for row in rows])


def log_error(error):
    if error == 0:
        return "-inf"
    return math.log10(error)


def naive_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def slope(xs, errors):
    return float(np.polyfit(np.log10(xs), np.log10(errors), 1)[0])


def cos_direct(x):
    return (1 - math.cos(x)) / (x * x)


def cos_stable(x):
    return 2 * math.sin(x / 2) ** 2 / (x * x)


def cos_reference(x):
    return (1 - mp.cos(x)) / x**2


def sqrt_direct(x):
    return math.sqrt(x + 1) - math.sqrt(x)


def sqrt_stable(x):
    return 1 / (math.sqrt(x + 1) + math.sqrt(x))


def sqrt_reference(x):
    return mp.sqrt(x + 1) - mp.sqrt(x)


def log_direct(x):
    return math.log(1 + x)


def log_reference(x):
    return mp.log(1 + x)


def task_1():
    # cond = |x * f'(x) / f(x)|
    x = 1.001
    y = 1.000001
    table(("f(x)", "x", "cond", "Оценка"), [
        ("sqrt(x)", "1e-8", "1/2 = 0.5", "хорошая"),
        ("ln(x)", x, f"1/|ln(x)| = {1 / abs(math.log(x)):.6f}", "плохая"),
        ("x - 1", y, f"|x/(x - 1)| = {1_000_001}", "плохая"),
        ("sin(x)", "pi/2", "|x*cot(x)| = 0", "хорошая"),
    ])
    print("Порядок по возрастанию cond: sin(x), sqrt(x), ln(x), x - 1.")
    print("У x - 1 ошибка входа усиливается примерно в миллион раз.")


def task_2():
    a, b, da = 1 + 1e-8, 1.0, 1e-9
    before, after = a - b, (a + da) - b
    cond_a, cond_b = abs(a / before), abs(b / before)
    print("df/da = 1, df/db = -1, поэтому cond_a = |a/(a-b)| и cond_b = |b/(a-b)|.")
    table(("cond_a", "cond_b", "Delta f / delta a", "Отн. сдвиг", "Прогноз"),
          [(cond_a, cond_b, (after - before) / da, (after - before) / before,
            cond_a * abs(da / a))])
    print("При близких a и b оба cond велики.")
    print("Абсолютный сдвиг близок к delta a, относительный к предсказанным 10%.")


def task_3():
    rows = []
    previous = None
    for eps in (1e-2, 1e-4, 1e-6):
        c = 1 - eps
        offset = math.sqrt(1 - c)
        left, right = 1 - offset, 1 + offset
        shift = abs(right - 1)
        rows.append((eps, left, right, shift, previous / shift if previous else "-"))
        previous = shift
    table(("eps", "Левый корень", "Правый корень", "Сдвиг", "Пред. / текущий"), rows)
    print("Из (x - 1)^2 = eps получаю сдвиг sqrt(eps).")
    print("При eps в 100 раз меньше сдвиг кратного корня меньше лишь в 10 раз.")


def task_4():
    rows = []
    for x in (1.0, 1e100, 1e154, 1e155, 1e200, 1e300):
        direct = x / math.sqrt(x * x + 1)
        stable = x / math.hypot(x, 1)
        exact_x = mp.mpf(x)
        reference = exact_x / mp.sqrt(exact_x * exact_x + 1)
        rows.append((x, direct, stable, mp.nstr(reference, 18)))
    table(("x", "Прямой", "Через hypot", "Эталон"), rows)
    print("Для x/sqrt(x*x + 1) число обусловленности 1/(1 + x*x) не больше 1.")
    print("Прямая формула переполняется при возведении в квадрат, а hypot сохраняет результат.")


def task_5():
    rows = []
    for b in (1e4, 1e8, 1e12):
        sqrt_disc = math.sqrt(b * b - 4)
        direct = (-b + sqrt_disc) / 2
        stable = 2 / (-b - sqrt_disc)
        exact_b = mp.mpf(b)
        reference = (-exact_b + mp.sqrt(exact_b * exact_b - 4)) / 2
        direct_error = relative_error(direct, reference)
        stable_error = relative_error(stable, reference)
        rows.append((b, direct, stable, mp.nstr(reference, 18), direct_error, stable_error))
    table(("b", "Прямой", "Устойчивый", "Эталон", "Ошибка прямого", "Ошибка устойчивого"), rows)
    catastrophic = next(row[0] for row in rows if row[4] >= 1)
    print(f"При b = 1e8 прямая формула ошибается на {rows[1][4]:.1%}, "
          f"а при b = {catastrophic:.0e} уже дает ноль.")
    print("В знаменателе числа одного знака, поэтому малый корень сохраняется.")


def task_6():
    rows = compare(("1e-2", "1e-4", "1e-6", "1e-8", "1e-10"),
                   cos_direct, cos_stable, cos_reference)
    error_table(rows)
    log_rows = []
    for x, _, _, _, direct_error, stable_error in rows:
        log_rows.append((math.log10(x), log_error(direct_error), log_error(stable_error)))
    table(("log10(x)", "log10(ошибка прямого)", "log10(ошибка устойчивого)"), log_rows)
    first = next(row[0] for row in rows if row[4] > 0.01)
    print(f"Ошибка прямой формулы впервые выше 1% при x = {first:.0e} из проверенных.")
    print("Когда cos(x) близок к 1, заменяю разность формулой с sin(x/2).")


def task_7():
    rows = compare(("1e4", "1e8", "1e12", "1e14", "1e16"),
                   sqrt_direct, sqrt_stable, sqrt_reference)
    error_table(rows)
    exponent = slope([row[0] for row in rows], [row[4] for row in rows])
    print(f"На этих точках ошибка прямой формулы растет примерно как x^{exponent:.2f}.")
    print("При больших x корни близки, а сопряженное убирает вычитание.")


def task_8():
    rows = compare(("1e-2", "1e-8", "1e-12", "1e-16"), log_direct, math.log1p, log_reference)
    error_table(rows)
    print("Справка: https://docs.python.org/3/library/math.html#math.log1p")
    print("log1p использует x напрямую, поэтому малая добавка сохраняется.")
    print("log1p, expm1 и hypot помогают сохранить точность при крайних значениях.")


def task_9():
    rows = []
    log_rows = []
    first_failure = None
    for n in (10**5, 10**6, 10**7):
        reference = mp.digamma(n + 1) + mp.euler
        terms = range(1, n + 1)
        loop_total = naive_sum(1 / k for k in terms)
        sum_total = sum(1 / k for k in terms)
        fsum_total = math.fsum(1 / k for k in terms)
        loop_error = relative_error(loop_total, reference)
        sum_error = relative_error(sum_total, reference)
        fsum_error = relative_error(fsum_total, reference)
        rows.append((n, loop_error, sum_error, fsum_error))
        log_rows.append((math.log10(n), log_error(loop_error), log_error(sum_error), log_error(fsum_error)))
        if first_failure is None and loop_error > 1e-10:
            first_failure = n
    table(("n", "Цикл", "sum", "fsum"), rows)
    table(("log10(n)", "log10(цикл)", "log10(sum)", "log10(fsum)"), log_rows)
    exponent = slope([row[0] for row in rows], [row[1] for row in rows])
    print(f"Наклон ошибки цикла: {exponent:.2f}.")
    if first_failure is None:
        print("На этой сетке ошибка ниже 1e-10.")
    else:
        print(f"Порог 1e-10 впервые превышен при n = {first_failure}.")


def task_10():
    x = 1e-4
    reference = cos_reference(mp.mpf("1e-4"))
    x32 = np.float32(x)
    result32 = (np.float32(1) - np.cos(x32)) / (x32 * x32)
    direct_error = relative_error(cos_direct(x), reference)
    rows = [
        ("x: 1e-2 -> 1e-4", relative_error(cos_direct(1e-2), cos_reference(mp.mpf("1e-2"))),
         direct_error),
        ("float64 -> float32", direct_error, relative_error(result32, reference)),
        ("прямой -> устойчивый", direct_error, relative_error(cos_stable(x), reference)),
    ]
    table(("Фактор", "До", "После", "log10(после/до)"),
          [(name, before, after, math.log10(after / before)) for name, before, after in rows])
    print("При малом x теряются цифры при вычитании, а float32 грубее округляет cos(x).")
    print("Формула с синусом убирает вычитание и снижает ошибку.")


def task_11():
    x = 1e-16
    first = math.log(1 + x)
    second = math.log1p((1 + x) - 1)
    reference = log_reference(mp.mpf("1e-16"))
    print(f"math.log(1 + x) = {first}, math.log1p((1 + x) - 1) = {second}, ==: {first == second}")
    print(f"Эталон: {mp.nstr(reference, 20)}, ошибки: {relative_error(first, reference):.2%}, "
          f"{relative_error(second, reference):.2%}")
    print("В обоих способах x теряется при сложении 1 + x.")
    print("Результаты совпали, а эталон показал ошибку у обоих.")


def task_12():
    rows = compare(tuple(f"1e{k}" for k in range(2, 17)), sqrt_direct, sqrt_stable, sqrt_reference)
    x = np.array([row[0] for row in rows])
    direct = np.array([row[4] for row in rows])
    stable = np.array([row[5] for row in rows])
    # Последняя точка уже уперлась в полную потерю результата.
    fit = (direct > 0) & (direct < 1)
    exponent = slope(x[fit], direct[fit])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.loglog(x, direct, "o-", label="Прямой")
    ax.loglog(x, stable, "s-", label="Через сопряженное")
    ax.loglog(x, np.finfo(float).eps * x, "--", label="eps * x")
    ax.set(xlabel="x", ylabel="Относительная ошибка", title="ДЗ 1, задача 12. Разность корней")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(Path(__file__).with_name("dz1_sqrt_error.png"), dpi=140)
    plt.close(fig)
    print(f"Наклон до потери результата: {exponent:.2f}, по оценке eps * x ожидается 1.")
    print("По сетке видны рост ошибки и разброс, по одной точке наклон не найти.")


def random_sums(seed):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    results = []
    for name, values in [
        ("random", [random.random() - 0.5 for _ in range(10**5)]),
        ("NumPy", (rng.random(10**5) - 0.5).tolist()),
    ]:
        results.append((name, sum(values), math.fsum(values), float(np.sum(values))))
    return results


def task_13():
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, seed 42 и 43")
    print("Выборка: 100000 чисел из [-0.5, 0.5).")
    first, repeat, changed = random_sums(42), random_sums(42), random_sums(43)
    rows = []
    for seed, results in ((42, first), (43, changed)):
        for name, builtin, accurate, numpy_total in results:
            rows.append((seed, name, repr(builtin), repr(accurate), repr(numpy_total),
                         builtin - accurate, numpy_total - accurate))
    table(("seed", "Генератор", "sum", "fsum", "numpy.sum", "sum - fsum", "numpy.sum - fsum"), rows)
    print(f"Повтор с seed 42 совпал: {first == repeat}.")
    print("При обоих seed расхождение numpy.sum с fsum порядка 1e-14.")
    print("Фиксирую seed и версии, чтобы повторить опыт.")


def task_14():
    cases = [
        ("Большие", [1e300, 1e300]),
        ("Малые", [1e-300, 1e-300]),
        ("Нули", [0.0, 0.0]),
        ("Денормализованные", [1e-320, 1e-320]),
        ("Малое с большим", [1e-300, 1e300, -1e300, 0.0, 1e-320]),
        ("inf", [1e300, math.inf]),
        ("inf и -inf", [math.inf, -math.inf]),
        ("nan", [1.0, math.nan, 2.0]),
        ("Все вместе", [1e300, 1e-300, 0.0, 1e-320, math.inf, math.nan]),
        ("Переполнение", [1e308, 1e308, -1e308]),
    ]
    rows = []
    for name, values in cases:
        try:
            accurate = math.fsum(values)
        except (ValueError, OverflowError) as error:
            accurate = type(error).__name__
        rows.append((name, naive_sum(values), accurate))
    table(("Данные", "Цикл", "fsum"), rows)
    print("В цикле NaN появляется сразу, fsum тоже возвращает NaN.")
    print("Один inf сохраняется, а два inf разных знаков дают в fsum ошибку.")
    print("Перед сложением проверяю math.isfinite и размер чисел.")


def task_15():
    print("Цель: проверить точность ln(1 + x) на малых положительных x.")
    print("Постановка: log(1 + x) и log1p(x) на сетке от 1e-1 до 1e-16 с шагом в порядок.")
    print("Эталон: mpmath.log(1 + x) с 50 десятичными знаками, x передаю строкой.")
    rows = compare(tuple(f"1e-{k}" for k in range(1, 17)),
                   log_direct, math.log1p, log_reference)
    table(("x", "Прямой", "Устойчивый", "Эталон", "Ошибка прямого", "Ошибка устойчивого"),
          [(x, repr(first), repr(second), expected, error1, error2)
           for x, first, second, expected, error1, error2 in rows])
    last_ok = next(row[0] for row in reversed(rows) if row[4] <= 0.01)
    max_error = max(row[5] for row in rows)
    print("При x = 1e-16 добавка пропадает из-за округления 1 + x.")
    print(f"Прямая формула дает до 1% ошибки вплоть до x = {last_ok:.0e}, "
          f"у log1p максимум {max_error:.2e}.")


def task_16():
    print("Источник: https://github.com/scipy/scipy/issues/11465")
    print("Строка из примера: log_1p = np.log(r) - np.log(mu + r)")
    r, mu = 1.0, 1e-16
    direct = np.log(r) - np.log(mu + r)
    stable = -np.log1p(mu / r)
    reference = -mp.log(1 + mp.mpf("1e-16"))
    table(("Прямой", "Через log1p", "Эталон", "Ошибка прямого", "Ошибка устойчивого"),
          [(float(direct), float(stable), float(reference),
            relative_error(direct, reference), relative_error(stable, reference))])
    print("При малом mu сумма округляется до r, поэтому разность равна нулю.")
    print("Формула -log1p(mu/r) сохраняет малую добавку.")


def main():
    tasks = (
        ("Число обусловленности", task_1), ("Вычитание близких чисел", task_2),
        ("Кратный корень", task_3), ("Обусловленность и устойчивость", task_4),
        ("Квадратное уравнение", task_5), ("Разность с косинусом", task_6),
        ("Разность корней", task_7), ("Логарифм", task_8),
        ("Гармонический ряд", task_9), ("Источники ошибки", task_10),
        ("Проверка эталоном", task_11), ("Логарифмическая сетка", task_12),
        ("Воспроизводимость", task_13), ("Границы float", task_14),
        ("Мини-отчет", task_15), ("Бонус. Пример из SciPy", task_16),
    )
    with mp.workdps(50):
        for number, (title, solve) in enumerate(tasks, 1):
            print(f"\nЗадача {number}. {title}")
            solve()


if __name__ == "__main__":
    main()
