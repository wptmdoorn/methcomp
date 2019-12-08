from methcomp import passingbablok
import matplotlib.pyplot as plt

method1 = [1, 2, 3, 4,
           5, 6, 7, 8,
           9, 10, 11, 12,
           13, 14, 15, 16,
           17, 18, 19, 20]
method2 = [1.03, 2.05, 2.79, 3.67,
           5.00, 5.82, 7.16, 7.69,
           8.53, 10.38, 11.11, 12.17,
           13.47, 13.83, 15.15, 16.12,
           16.94, 18.09, 19.13, 19.54]

passingbablok(method1, method2, CI=.95)
plt.show()