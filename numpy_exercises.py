import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# 1.
a = np.arange(24).reshape(6, 4)
print("Задание 1")
print(a)
print("Форма:", a.shape)
print("Измерений:", a.ndim)
print("Элементов:", a.size)
print("Тип:", a.dtype)

# 2.
b = a.reshape(4, 6)
c = b.reshape(2, 3, 4)
d = c.reshape(24)

print("\nЗадание 2")
print("Новые формы:", b.shape, c.shape, d.shape)
print("Числа сохранились:")
print(np.array_equal(a.ravel(), b.ravel()))
print(np.array_equal(a.ravel(), c.ravel()))
print(np.array_equal(a.ravel(), d))

# 3.
print("\nЗадание 3")
column = a[:, 2]
print("До изменения:", a[0, 2])
column[0] = 999
print("После изменения:", a[0, 2])
print("Это view: исходный массив тоже изменился.")

# Возвращаю число обратно
column[0] = 2

# 4.
print("\nЗадание 4")
numbers = a[a > 10]
print("Числа больше 10:", numbers)
print("Их среднее:", numbers.mean())

# 5.
print("\nЗадание 5")
row_means = a.mean(axis=1, keepdims=True)
result = a - row_means
print(result)
print("Форма:", result.shape)
print("Суммы строк:", result.sum(axis=1))
print("Форма сохранилась:", result.shape == a.shape)
print("Суммы равны нулю:", np.allclose(result.sum(axis=1), 0))
