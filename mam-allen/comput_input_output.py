import matplotlib.pyplot as plt
import json
import os
from config import data_path

with open(os.path.join(data_path,"rate_list.json"), 'r') as file:
    rate_list = json.load(file)

for pop in pop_list:
    plt.plot(factor, np.array(rate_list[pop]),label = f"rate_{pop}")
    plt.legend()
    plt.title('Relationship between input and rate')
    plt.xlabel('Factor')
    plt.ylabel('rate')
    plt.savefig(os.path.join(data_path,"input_factor-rate_{pop}.png".format(pop)))