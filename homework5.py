import time

import numpy as np


def loop_version(mat):
    n, k = mat.shape
    row_sums = [0.0] * n
    for i in range(n):
        for j in range(k):
            row_sums[i] += mat[i, j]

    col_max = [mat[0, j] for j in range(k)]
    col_arg = [0] * k
    for j in range(k):
        for i in range(1, n):
            if mat[i, j] > col_max[j]:
                col_max[j] = mat[i, j]
                col_arg[j] = i
    return row_sums, col_max, col_arg


def numpy_version(mat):
    row_sums = mat.sum(axis=1)
    col_max = mat.max(axis=0)
    col_arg = mat.argmax(axis=0)
    return row_sums, col_max, col_arg


def measure(fn, mat):
    best = float("inf")
    result = None
    for _ in range(3):
        start = time.perf_counter()
        result = fn(mat)
        best = min(best, time.perf_counter() - start)
    return result, best


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    m = rng.normal(size=(200_000, 5))
    r_loop, t_loop = measure(loop_version, m)
    r_numpy, t_numpy = measure(numpy_version, m)

    print(f"Цикл: {t_loop * 1000:.2f} мс")
    print(f"NumPy: {t_numpy * 1000:.2f} мс")
    print(f"Ускорение: {t_loop / t_numpy:.1f} раз")
    print()
    print("Суммы строк совпадают:", np.allclose(r_loop[0], r_numpy[0]))
    print("Максимумы совпадают:", np.array_equal(r_loop[1], r_numpy[1]))
    print("Индексы совпадают:", np.array_equal(r_loop[2], r_numpy[2]))
    print()
    print("Максимумы:", np.round(r_numpy[1], 4))
    print("Индексы:", r_numpy[2])
