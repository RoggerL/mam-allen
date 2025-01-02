
with open(os.path.join(data_path,"rate_list.json"), 'r') as file:
    rate_list = json.load(file)
    
layer_types = ["23","4","5","6"]
neuron_types = ["E","S","P","H"]
for layer_type in layer_types:
    plt.figure()
    for neuron_type in neuron_types:
        plt.subplot(2, 2, i + 1)  # 创建2x2的子图
        pop = neuron_type+layer_type
        plt.plot(factor, np.array(rate_list[pop]),label = f"rate_{pop}")
        plt.legend()
        plt.title('Relationship between input and rate')
        plt.xlabel('Factor')
        plt.ylabel('rate')
        
    plt.savefig(os.path.join(data_path, f"input_factor-rate_{layer_type}.png"))
    plt.close()  # 关闭图像，避免重叠绘图    