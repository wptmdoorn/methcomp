# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from methcomp import Linear, linear

method1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
method2 = [
    1.03,
    2.05,
    2.79,
    3.67,
    5.00,
    5.82,
    7.16,
    7.69,
    8.53,
    10.38,
    11.11,
    12.17,
    13.47,
    13.83,
    15.15,
    16.12,
    16.94,
    18.09,
    19.13,
    19.54,
]
CI = 0.95
x_label = "$M_1$"
y_label = "$M_2$"
# Make subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Legacy method
linear(
    method1,
    method2,
    CI=CI,
    ax=axs[0],
    square=True,
    x_label=x_label,
    y_label=y_label,
    title="Linear: Legacy",
)

# Regressor method - preferred approach
Linear(method1, method2, CI=CI).plot(
    ax=axs[1], square=True, x_label=x_label, y_label=y_label, title="Linear: Regressor"
)


plt.show()
