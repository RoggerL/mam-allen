import numpy as np
import analysis_helpers as ah

# 生成一个 3x100 的矩阵，元素为0或1，1的概率为0.8
spike_data = np.random.choice([0, 1], size=(100, 100), p=[0.2, 0.8])

# 调用 pop_rate 函数，将 spike_data 传递给它
rate = ah.pop_rate(spike_data, 0., 1000., 100,dt = 1.)

print(rate)

