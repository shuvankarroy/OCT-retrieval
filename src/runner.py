# Runner script for complete automation
import os
# dataset_name = "Duke-AMD-DME", 
dataset_name = ["Duke-AMD-DME-Normal", "Duke-AMD-Normal", "Duke-DME-Normal"]
j = 0
for dataset in dataset_name:
    for i in range(0, 3):
        print(dataset, i, i+1)
        os.system(f"siamese_net_avg.py {dataset} {i} {i+1}")

    