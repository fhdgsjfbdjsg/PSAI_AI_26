import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. ДАННЫЕ
# ==============================

X = np.array([
    [3, 6],
    [-3, 6],
    [3, -6],
    [-3, -6]
], dtype=float)

E = np.array([0, 0, 0, 1], dtype=float)

# Нормализация (чтобы не было overflow)
X = X / np.max(np.abs(X))

eta = 0.01
epochs = 100


# ==============================
# 2. ОБУЧЕНИЕ
# ==============================

def train_perceptron(X, E, eta, epochs):
    np.random.seed(42)
    w = np.random.randn(2) * 0.1
    T = np.random.randn() * 0.1

    for _ in range(epochs):
        for x, target in zip(X, E):
            S = np.dot(w, x) - T
            error = target - S

            # ограничение ошибки (защита от разлёта)
            error = np.clip(error, -10, 10)

            w += eta * error * x
            T -= eta * error

    return w, T


w, T = train_perceptron(X, E, eta, epochs)


# ==============================
# 3. ИНТЕРАКТИВНАЯ КЛАССИФИКАЦИЯ
# ==============================

print("Чтобы выйти, введите 'exit'")

while True:

    try:
        x1 = input("\nx1 = ")
        if x1.lower() == 'exit':
            break

        x2 = input("x2 = ")
        if x2.lower() == 'exit':
            break

        x1 = float(x1)
        x2 = float(x2)

        # нормализуем так же как при обучении
        x_input = np.array([x1, x2]) / 6.0

        S = np.dot(w, x_input) - T
        y = 1 if S > 0 else 0

        print("\n========== РЕЗУЛЬТАТ КЛАССИФИКАЦИИ ==========")
        print(f"Введённый вектор: ({x1:.1f}, {x2:.1f})")
        print(f"Взвешенная сумма S = {S:.6f}")
        print(f"Класс сети: {y}")
        print("==============================================")

    except:
        print("Ошибка ввода. Попробуйте снова.")