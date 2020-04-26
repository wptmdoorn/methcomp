from methcomp import seg, segscores
import matplotlib.pyplot as plt
from random import uniform

ref = [uniform(2, 18) for _ in range(300)]
test = [uniform(2, 18) for _ in range(300)]

zones = segscores(ref, test, units='mmol')
print('SEG scores: {}'.format(zones))

seg(ref, test, units='mmol', percentage=False)
plt.show()
