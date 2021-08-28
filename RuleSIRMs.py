# DeepSIRMsのルール部分記述
# 二層です

import numpy as np
import tensorflow as tf

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
    
    @tf.function
    def h1(self):
        h1 = np.zeros((self.n, self.p))
        for i in range(self.p):
            h1[i,:] = 1 - np.abs(self.a1[i,:]-self.X[i]) / self.b1[i,:]
        return h1

    @tf.function
    def first_layer(self, h1):
        # 一層目出力
        h1 = self.h1()
        y1 = np.dot(np.sum(h1 * self.y1, axis=1) / np.sum(h1,axis=1), self.w1)
        return y1

    def second_layer(self,y1):
        # 二層目出力
        a2 = np.append(self.a1, [[0.0, 0.5, 1.0]], axis=0)
        b2 = np.append(self.b1, [[1.0, 0.5, 1.0]], axis=0)
        y2 = np.append(self.y1, [[0.4, 0.4, 0.4]], axis=0)
        w2 = np.append(self.w1, 0.7)
        h2_add = 1 - np.abs(a2[self.p, :]-y1) / b2[self.p,:]
        h2 = np.append(self.h1, [h2_add], axis=0)

        # 二層目出力 
        y2 = np.dot(np.sum(h2 * y2, axis=1) / np.sum(h2, axis=1), w2)
        return y2
    


