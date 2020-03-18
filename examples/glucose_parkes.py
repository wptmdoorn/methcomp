from methcomp import parkes, parkeszones
import matplotlib.pyplot as plt
from random import uniform

ref = [uniform(1, 30) for _ in range(300)]
test = [uniform(1, 30) for _ in range(300)]

zones = parkeszones(1, ref, test, units='mmol', numeric=True)
print('Clarkes zones: {}'.format(zones))

parkes(1, ref, test, units='mmol', color_points='auto', percentage=True)
plt.show()
