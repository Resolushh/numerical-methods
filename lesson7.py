from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path(__file__).resolve().parent


def rotation(degrees):
    angle = np.deg2rad(degrees)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]])


def hexagon():
    angles = 2 * np.pi * np.arange(6) / 6
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.vstack((points, points[0]))

    rotated = points @ rotation(45).T
    reflection = np.array([[0, 1], [1, 0]])
    reflected = points @ reflection.T

    print("1. Шестиугольник")
    print("Первая вершина:", points[0])
    print("Поворот на 45°:", np.round(rotated[0], 4))
    print("Отражение по y = x:", reflected[0])
    print("Определитель отражения:", round(np.linalg.det(reflection)))

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.plot(*points.T, "o-", color="gray", label="Исходный")
    ax.plot(*rotated.T, "s-", color="tab:blue", label="Поворот на 45°")
    ax.plot(*reflected.T, "^--", color="tab:red", label="Отражение по y = x")
    ax.plot([-1.15, 1.15], [-1.15, 1.15], ":", color="gray", alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set(xlabel="x", ylabel="y", aspect="equal")
    ax.legend(loc="upper right", bbox_to_anchor=(1.06, 1.12))
    fig.savefig(OUTPUT / "hexagon.png", dpi=140)
    plt.close(fig)


def cloud():
    rng = np.random.default_rng(42)
    points = rng.uniform(-2, 2, size=(200, 2))
    points_h = np.column_stack((points, np.ones(len(points))))

    scale = np.diag([1.5, 0.5, 1])
    rotate = np.eye(3)
    rotate[:2, :2] = rotation(60)
    move = np.array([[1, 0, 1], [0, 1, -1], [0, 0, 1]])
    changed_h = points_h @ (move @ rotate @ scale).T

    undo_move = np.array([[1, 0, -1], [0, 1, 1], [0, 0, 1]])
    undo_rotate = np.eye(3)
    undo_rotate[:2, :2] = rotation(-60)
    undo_scale = np.diag([1 / 1.5, 1 / 0.5, 1])
    restored = (changed_h @ (undo_scale @ undo_rotate @ undo_move).T)[:, :2]
    wrong = (changed_h @ (undo_move @ undo_rotate @ undo_scale).T)[:, :2]

    print("\n2. Облако из 200 точек")
    print("Обратный порядок:", np.allclose(restored, points, atol=1e-12, rtol=0))
    print(f"Максимальная разница: {np.max(np.abs(restored - points)):.2e}")
    print("Прямой порядок:", np.allclose(wrong, points, atol=1e-12, rtol=0))
    print(f"Максимальная разница: {np.max(np.abs(wrong - points)):.2f}")

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.scatter(*points.T, s=16, color="gray", alpha=0.65, label="Исходные")
    ax.scatter(*changed_h[:, :2].T, s=16, color="tab:blue", alpha=0.65, label="Преобразованные")
    ax.scatter(*restored.T, s=38, facecolors="none", edgecolors="tab:red",
               linewidths=0.75, label="Восстановленные")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set(xlabel="x", ylabel="y", aspect="equal")
    ax.legend()
    fig.savefig(OUTPUT / "cloud.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    hexagon()
    cloud()
