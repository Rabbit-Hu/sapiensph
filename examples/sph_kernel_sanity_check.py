import numpy as np


def W(r, h):
    q = np.linalg.norm(r) / h
    if q < 1.0:
        return 1.0 / (np.pi * h ** 3.0) * (1.0 - 1.5 * q ** 2.0 + 0.75 * q ** 3.0)
    elif q < 2.0:
        return 1.0 / (np.pi * h ** 3.0) * 0.25 * (2.0 - q) ** 3.0
    else:
        return 0.0
    
def dWdr(r, h):
    q = np.linalg.norm(r) / h
    if q < 1.0:
        return 1.0 / (np.pi * h ** 4.0) * (-3.0 * q + 2.25 * q ** 2.0) * r / np.linalg.norm(r)
    elif q < 2.0:
        return 1.0 / (np.pi * h ** 4.0) * -0.75 * (2.0 - q) ** 2.0 * r / np.linalg.norm(r)
    else:
        return np.array([0.0, 0.0, 0.0])

h = 1e-2
dt = 1e-3
m = 1e3 * h**3.0
xs = np.linspace(-1, 1, 3) * h
positions = np.array(np.meshgrid(xs, xs, xs)).reshape(3, -1).T
n = len(positions)
i = n // 2
rho = 0.0
for j in range(n):
    rho += W(positions[i] - positions[j], h) * m
print(f"rho = {rho}")

sum_w = np.array([0.0, 0.0, 0.0])
sum_w_dot_w = 0.0
for j in range(n):
    if i == j:
        continue
    w = dWdr(positions[i] - positions[j], h)
    sum_w += w
    sum_w_dot_w += np.dot(w, w)
delta_beta = 1.0 / (np.dot(sum_w, sum_w) + sum_w_dot_w)
beta = dt**2.0 * m**2.0 * 2.0 / rho**2.0

print(f"delta_beta = {delta_beta}")
print(f"beta = {beta}")
print(f"delta = {delta_beta / beta}")