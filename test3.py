# 全自動ファジィルール作成器 完成！！！

import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

input_size = 4
part_num = 3

a = np.linspace(0, 1, part_num)
for i in range(input_size-1): a = np.append(a, np.linspace(0, 1, part_num), axis=0)
a = a.reshape(input_size, part_num)[:input_size-1,:]
#print(a)

class DeepSIRMs:
    l = 2 # 層の数
    def __init__(self, input_size, part_num):
        self.n = input_size
        self.p = part_num

        a = np.linspace(0, 1, self.p)
        b = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n): 
            a = np.append(a, np.linspace(0, 1, self.p), axis=0)
            b = np.append(b, np.ones(self.p) / (self.p**1.5), axis=0)
        self.a1 = a.reshape(self.n+1, self.p)[:self.n,:]
        self.b1 = b.reshape(self.n+1, self.p)[:self.n,:]
        self.a2 = a.reshape(self.n+1, self.p)
        self.b2 = b.reshape(self.n+1, self.p)

        self.y1 = np.random.normal(0.5, 0.3, (self.n, self.p))
        self.w1 = np.random.normal(0.5, 0.3, (self.n))
        self.y2 = np.random.normal(0.5, 0.3, (self.n+1, self.p))
        self.w2 = np.random.normal(0.5, 0.3, (self.n+1))

    def print_1(self):
        print(self.a1)
        print(self.b1)
        print(self.a2)
        print(self.b2)

    def print_2(self):
        print(self.y1)
        print(self.w1)
        print(self.y2)
        print(self.w2)
    

aaa = DeepSIRMs(5,2)
aaa.print_1()
aaa.print_2()