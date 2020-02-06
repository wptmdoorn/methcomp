from methcomp import clarke
import matplotlib.pyplot as plt
from random import uniform

ref = [uniform(5, 10)for _ in range(50)]
test = [uniform(5, 10) for _ in range(50)]

clarke(ref, test, units='mmol')
plt.show()