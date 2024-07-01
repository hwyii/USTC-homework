import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import davies_bouldin_score
import os

import os

# 修改工作目录
new_directory = 'D:\Study\MachineLearning\ML_lab\lab4_Clustering'
os.chdir(new_directory)

Aggregation = pd.read_table("Aggregation.txt",sep = ' ',header=None)
D31 = pd.read_table("D31.txt",sep = ' ',header=None)
R15 = pd.read_table("R15.txt",sep = ' ',header=None)
print(Aggregation)