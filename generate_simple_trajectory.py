import os
import numpy as np
import matplotlib.pyplot as plt

# 函数：生成随机二次函数参数a、b和c
def generate_coefficients():
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-10, 10)
    return a, b

# 函数：计算函数值
def func(x, a, b):
    return (a * x - b) ** 2

# 函数：计算梯度
def gradient(x, a, b):
    return 2 * (a ** 2) * x - 2 * a * b

def perform_gradient_descent(threshold, learning_rate, a, b):
    x = np.random.rand()
    optimization_path = []

    while True:
        grad = gradient(x, a, b)
        x -= learning_rate * grad
        y = func(x, a, b)

        optimization_path.append((x, y, grad))

        if abs(grad) < threshold:
            break

    return optimization_path


np.random.seed(0)

parent_folder = "data"
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)

learning_rate = 0.01
threshold = 0.001

# 开始求解100个二次函数优化问题
for i in range(100):
    a, b = generate_coefficients()

    optimization_path = perform_gradient_descent(threshold, learning_rate, a, b)

    iter_folder = os.path.join(parent_folder, f"iter_{i}")
    os.makedirs(iter_folder)

    np.savetxt(os.path.join(iter_folder, "parameters.txt"), np.array([a, b]))
    np.savetxt(os.path.join(iter_folder, "x_values.txt"), [x[0] for x in optimization_path])
    np.savetxt(os.path.join(iter_folder, "y_values.txt"), [x[1] for x in optimization_path])
    np.savetxt(os.path.join(iter_folder, "x_gradients.txt"), [x[2] for x in optimization_path])

    # 绘制优化轨迹
    plt.plot([x[1] for x in optimization_path], '-o')
    plt.xlabel("Iteration")
    plt.ylabel("y")
    plt.title(f"Optimization Path")

    plt.savefig(os.path.join(iter_folder, "optimization_path.png"))
    plt.close()

    # 绘制梯度轨迹
    plt.plot([x[2] for x in optimization_path], 'r-o')
    plt.xlabel("Iteration")
    plt.ylabel("Gradient")
    plt.title(f"Gradient Path")

    plt.savefig(os.path.join(iter_folder, "gradient_path.png"))
    plt.close()
