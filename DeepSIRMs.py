# DeepSIRMsの一般化コードです
# 二層です

import numpy as np

class DeepSIRMs:
    l = 2 # 層の数
    def __init__(self, X, p, a1, b1, y1, w1):
        self.X = X 
        self.n = np.shape(X)[0]  
        self.p = p
        self.a1 = a1
        self.b1 = b1
        self.h1 = np.zeros((self.n, p))
        self.y1 = y1
        self.w1 = w1
    
    

    def first_layer(self):
        # 一層目出力
        for i in range(self.p):
            self.h1[i,:] = 1 - np.abs(self.a1[i,:]-self.X[i]) / self.b1[i,:] #grade cal 
        y1 = np.dot(np.sum(self.h1 * self.y1, axis=1) / np.sum(self.h1,axis=1), self.w1)
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



