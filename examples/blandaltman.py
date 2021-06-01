# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from methcomp import blandaltman

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

ba = blandaltman.BlandAltman(method1, method2, CI=0.95)
ba2 = blandaltman.BlandAltman(method1, method2, CI=0.95)

# get statistics
output = ba.calculate()
print(output)

# plot with some title
ba.plot(graph_title="Test")  # this directly uses the previous .compute() statement
plt.show()

ba2.plot(graph_title="Test 2")  # this calls .compute() internally to compute parameters
plt.show()
