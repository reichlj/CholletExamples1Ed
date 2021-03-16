from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

data_dir = Path(r'C:\data\jena_climate\jena_climate_2009_2016.csv')
with open(data_dir,encoding='utf-8') as f:
    data = f.read()
lines = data.split('\n')
header = lines[0].split(',')
header = [word[1:-1] for word in header]
lines = lines[1:]
print(header)
print(len(lines))

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.show()
plt.plot(range(1440), temp[:1440])
plt.show()