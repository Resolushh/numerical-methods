import timeit
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT = Path(__file__).resolve().parent


def membership():
    sizes = [100, 1000, 10000]
    checks = 1000
    list_times = []
    set_times = []

    print("1. Поиск числа -1")
    for n in sizes:
        numbers = list(range(n))
        unique = set(numbers)
        list_runs = timeit.repeat("-1 in numbers", globals={"numbers": numbers}, number=checks, repeat=5)
        set_runs = timeit.repeat("-1 in unique", globals={"unique": unique}, number=checks, repeat=5)
        list_us = min(list_runs) / checks * 1_000_000
        set_us = min(set_runs) / checks * 1_000_000
        list_times.append(list_us)
        set_times.append(set_us)
        print(f"n={n}: list {list_us:.3f} мкс, set {set_us:.3f} мкс")
        print(f"set быстрее в {list_us / set_us:.1f} раз")

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(sizes, list_times, "o-", label="list")
    ax.plot(sizes, set_times, "s-", label="set")
    ax.set(xlabel="Размер коллекции", ylabel="Время одной проверки, мкс", yscale="log")
    ax.set_xticks(sizes)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(OUTPUT / "membership.png", dpi=140)
    plt.close(fig)


def squares_append(n):
    result = []
    for i in range(n):
        result.append(i * i)
    return result


def squares_comprehension(n):
    return [i * i for i in range(n)]


def squares_map(n):
    return list(map(lambda i: i * i, range(n)))


def compare_squares():
    sizes = [100_000, 1_000_000]
    methods = [
        ("append", squares_append),
        ("Списочное включение", squares_comprehension),
        ("map + lambda", squares_map),
    ]
    positions = [0, 1]
    width = 0.24
    fig, (ax_time, ax_memory) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

    print("\n2. Список квадратов")
    for i, (name, make_squares) in enumerate(methods):
        times = []
        peaks = []
        for n in sizes:
            data = {"make_squares": make_squares, "n": n}
            runs = timeit.repeat("make_squares(n)", globals=data, number=1, repeat=5)
            time_ms = min(runs) * 1000

            tracemalloc.start()
            result = make_squares(n)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            del result

            peak_mb = peak / 1024**2
            times.append(time_ms)
            peaks.append(peak_mb)
            print(f"n={n}, {name}: {time_ms:.2f} мс, {peak_mb:.2f} МиБ")

        ax_time.plot(positions, times, "o-", label=name)
        bars = [position + (i - 1) * width for position in positions]
        ax_memory.bar(bars, peaks, width, label=name)

    labels = ["100 000", "1 000 000"]
    for ax in (ax_time, ax_memory):
        ax.set_xticks(positions, labels)
        ax.set_xlabel("Размер списка")
        ax.legend(fontsize=8)
    ax_time.set_ylabel("Время, мс")
    ax_memory.set_ylabel("Пиковая память, МиБ")
    fig.savefig(OUTPUT / "squares.png", dpi=140)
    plt.close(fig)
    print("Память почти одинаковая: во всех случаях получается список квадратов.")
    print("В списочном включении нет вызова append на каждом шаге.")
    print("У map есть вызов lambda для каждого числа.")


if __name__ == "__main__":
    membership()
    compare_squares()
