from decimal import Decimal, localcontext

import numpy as np


def task_1():
    a = np.arange(60).reshape(3, 4, 5)
    print("shape:", a.shape, "ndim:", a.ndim, "size:", a.size)
    print("dtype:", a.dtype, "strides:", a.strides)
    print("C_CONTIGUOUS:", a.flags.c_contiguous)


def task_2():
    a = np.array([1, 2, 3, 4])
    print("a / 2:", a / 2)
    print("a // 2:", a // 2)
    print("a / 2.0:", a / 2.0)
    print("Типы:", a.dtype, (a / 2).dtype, (a // 2).dtype)
    print("Исходный a:", a)
    b = np.array([1, 2, 3], dtype=np.int8)
    result = b + 127
    print("b + 127:", result, result.dtype)


def task_3():
    x32 = float(np.float32(0.1))
    x64 = float(np.float64(0.1))
    for name, value in (("float32", x32), ("float64", x64)):
        error = value - 0.1
        print(name, "разница:", repr(error), f"{error:.20e}")
    print("Отношение к нулевой разнице float64 не определено")
    with localcontext() as context:
        context.prec = 50
        exact = Decimal("0.1")
        error32 = abs(Decimal.from_float(x32) - exact)
        error64 = abs(Decimal.from_float(x64) - exact)
        print("Ошибка относительно точного 0.1:", error32, error64)
        print("Отношение ошибок:", error32 / error64)


def task_4():
    arrays = [
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float32),
        np.ones((2, 3), dtype=np.int16),
        np.full((2, 3), 7, dtype=np.uint8),
    ]
    for a in arrays:
        print(a.dtype, "байт:", a.nbytes)
    saved = 1000 * 1000 * (np.dtype("float64").itemsize - np.dtype("float32").itemsize)
    print("Экономия float32 на (1000, 1000):", saved, "байт")


def task_5():
    a = np.arange(24)
    current = a
    for shape in ((4, 6), (2, 3, 4), (2, 2, 6), (24,)):
        current = current.reshape(shape)
        print(current.shape, "данные сохранились:", np.array_equal(a, current.ravel()))


def task_6():
    m = np.arange(12).reshape(3, 4)
    f = m.flatten()
    r = m.ravel()
    print("flatten без base:", f.base is None)
    print("ravel с base:", r.base is not None)
    f[0] = 999
    print("После f[0] = 999, M[0, 0]:", m[0, 0])
    r[1] = 888
    print("После r[1] = 888:")
    print(m)


def task_7():
    v = np.arange(5)
    print("v[:, None]:", v[:, None].shape)
    print("v[None, :]:", v[None, :].shape)
    print("squeeze:", np.squeeze(v[:, None]).shape)
    transposed = np.arange(6).reshape(2, 3).T
    print("Транспонирование:", transposed.shape)
    print(transposed)
    result = v[:, None] + v[None, :]
    print("Сложение:", result.shape)
    print(result)


def task_8():
    a = np.arange(6).reshape(2, 3)
    b = a.T
    print("У B есть base:", b.base is not None)
    print("strides A:", a.strides, "strides B:", b.strides)
    b[0, 1] = 999
    print("A после изменения B:")
    print(a)


def task_9():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(6).reshape(3, 2)
    for name, result in (("A @ B", a @ b), ("A.T @ B.T", a.T @ b.T), ("A * A", a * a)):
        print(name, result.shape)
        print(result)
    try:
        print(a * b)
    except ValueError:
        print("A * B: формы (2, 3) и (3, 2) несовместимы")
    diagonal = np.diag([2, 3])
    print("Для диагональных матриц * и @ совпали:",
          np.array_equal(diagonal * diagonal, diagonal @ diagonal))


def task_10():
    m = np.arange(12).reshape(3, 4)
    print(m)
    print("Суммы столбцов:", m.sum(axis=0))
    print("Суммы строк:", m.sum(axis=1))
    print("Средние столбцов:", m.mean(axis=0))
    print("Средние строк:", m.mean(axis=1))
    print("Индексы максимумов в столбцах:", m.argmax(axis=0))
    print("Индексы максимумов в строках:", m.argmax(axis=1))
    print("Первый столбец:", m[:, 0], "сумма:", m[:, 0].sum())


def task_11():
    x = np.random.default_rng(42).normal(size=(5, 3))
    mean = x.mean(axis=0)
    mean_keep = x.mean(axis=0, keepdims=True)
    print("Формы средних:", mean.shape, mean_keep.shape)
    print("После вычитания:", (x - mean).shape, (x - mean_keep).shape)
    print("Формы std строк:", x.std(axis=1).shape, x.std(axis=1, keepdims=True).shape)
    normalized = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    print("Средние строк:", normalized.mean(axis=1))
    print("std строк:", normalized.std(axis=1))
    print("Средние близки к 0:", np.allclose(normalized.mean(axis=1), 0))
    print("std близки к 1:", np.allclose(normalized.std(axis=1), 1))


def task_12():
    a = np.arange(10)
    b = a[2:5]
    b[0] = 999
    print("После изменения среза:", a)
    a = np.arange(10)
    c = a[2:5].copy()
    c[0] = 999
    print("После изменения копии:", a)


def task_13():
    a = np.random.default_rng(0).normal(size=20)
    print("Положительных:", a[a > 0].size)
    print("Между -1 и 1:", a[(a > -1) & (a < 1)].size)
    print("Индексы 0, 5, 10, 15:", a[[0, 5, 10, 15]])
    print("Выбросы:", a[np.abs(a) > 2])
    try:
        print(a[a > 0] & (a < 1))
    except (TypeError, ValueError):
        print("a[a > 0] & (a < 1): ошибка")
    print("Правильная маска:", (a > 0) & (a < 1))
    a[a < 0] = 0
    print("Отрицательные заменены на 0:", a)


def task_14():
    x = np.random.default_rng(1).normal(size=(100, 3))
    normalized = (x - x.mean(axis=0)) / x.std(axis=0)
    print("Средние столбцов:", normalized.mean(axis=0))
    print("std столбцов:", normalized.std(axis=0))
    print("Средние близки к 0:", np.allclose(normalized.mean(axis=0), 0))
    print("std близки к 1:", np.allclose(normalized.std(axis=0), 1))
    d = np.sqrt(((normalized[:, None, :] - normalized[None, :, :]) ** 2).sum(axis=2))
    print("Форма D:", d.shape)
    print("Диагональ близка к 0:", np.allclose(d.diagonal(), 0))
    print("D симметрична:", np.allclose(d, d.T))


def task_15():
    raw = np.random.default_rng(42).normal(20, 2, size=(30, 4))
    raw[7, 2] = 100
    print("shape:", raw.shape, "dtype:", raw.dtype, "size:", raw.size)
    mask = np.abs(raw - raw.mean(axis=0)) > 3 * raw.std(axis=0)
    print("Выбросов:", mask.sum(), "позиции:", np.argwhere(mask))
    clean = raw.copy()
    clean[mask] = np.nan
    print("NaN:", np.isnan(clean).sum())
    clean = np.where(np.isnan(clean), np.nanmean(clean, axis=0), clean)
    print("Значение вместо выброса:", clean[7, 2])
    transposed = clean.T
    print("После .T:", transposed.shape)
    print("Обратно:", transposed.T.shape, np.array_equal(clean, transposed.T))
    flat = clean.ravel()
    restored = flat.reshape(30, 4)
    print("После ravel:", flat.shape)
    print("Обратно:", restored.shape, np.array_equal(clean, restored))
    print("Средние датчиков:", clean.mean(axis=0))
    print("std датчиков:", clean.std(axis=0))
    normalized = (clean - clean.mean(axis=0)) / clean.std(axis=0)
    print("После нормировки, средние:", normalized.mean(axis=0))
    print("После нормировки, std:", normalized.std(axis=0))
    print("Средние близки к 0:", np.allclose(normalized.mean(axis=0), 0))
    print("std близки к 1:", np.allclose(normalized.std(axis=0), 1))


if __name__ == "__main__":
    tasks = [task_1, task_2, task_3, task_4, task_5, task_6, task_7, task_8,
             task_9, task_10, task_11, task_12, task_13, task_14, task_15]
    for number, task in enumerate(tasks, 1):
        print(f"\nЗадача {number}")
        task()
