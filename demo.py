import numpy as np
a = []
b = np.zeros(5)
for i in range(3):
    a.append(b)
print(a)
a = np.array(a).T
print(a)
a = a.T
print(a)