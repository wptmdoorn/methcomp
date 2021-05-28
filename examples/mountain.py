# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from methcomp import mountain

# Synthetic data for 3 methods
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
method3 = [
    0.82,
    2.09,
    4.31,
    4.18,
    4.85,
    6.01,
    7.73,
    7.95,
    9.78,
    10.07,
    11.64,
    11.88,
    12.90,
    13.55,
    14.57,
    16.82,
    16.95,
    17.62,
    18.44,
    20.14,
]

# Build first mountain plot : compare method 1 and 2
# This will create the axis object as it's the first plot
mountain.Mountain(method1, method2, n_percentiles=500).plot(
    color="blue", label="$M_1$ - $M_2$", unit="ng/ml"
)
# Build second mountain plot : compare method 1 and 3
# This will reuse the previous axis object
mountain.Mountain(method1, method3, n_percentiles=500).plot(
    color="green", label="$M_1$ - $M_3$", unit="ng/ml"
)

plt.show()
