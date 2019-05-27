import numpy as np

diff = np.zeros([10,2])
sma = np.zeros([10,2])

appended = np.append(diff,sma,axis=1)

print(appended.shape)