# DeepSIRMsのトレーニング部分
# 二層です

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from RuleSIRMs import RuleSIRMs

import pandas as pd
csv_xor_input = pd.read_csv(filepath_or_buffer="./Dataset/XOR.csv", encoding="utf-8", sep=",")

fuzzy = RuleSIRMs()

iters_num = 10

train_loss_list = []
train_acc_list = []
test_acc_list = []

X_train = csv_xor_input.values[:,:2]
t = csv_xor_input.values[:,2]
print(X_train, t)

for i in range(iters_num):
    grad = fuzzy.gradient(X, T)