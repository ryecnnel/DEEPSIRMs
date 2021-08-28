import numpy as np
from numpy.core.fromnumeric import reshape
import tensorflow as tf
import contextlib

class DeepSIRMs:
    def __init__(self, input_size, part_num):
        self.n = input_size
        self.p = part_num

        a = np.linspace(0, 1, self.p)
        b = np.ones(self.p) / (self.p**1.5)
        for i in range(self.n): 
            a = np.append(a, np.linspace(0, 1, self.p), axis=0)
            b = np.append(b, np.ones(self.p) / (self.p**1.5), axis=0)
        self.x = tf.constant(x)
        self.a1 = tf.Variable(a.reshape(self.n+1, self.p)[:self.n,:])
        self.b1 = tf.Variable(b.reshape(self.n+1, self.p)[:self.n,:])

        self.y = tf.Variable(y)
        self.z = tf.Variable(z)
    
    @tf.function
    def first_layer(self, X):
        # 1層目ルール群適合度
        h1 = np.zeros((self.n, self.p))

        for i in range(self.n):
            # h1[i,:] = 1 - np.abs(self.a1[i,:]-X[i]) / self.b1[i,:]
            h1[i,:] = tf.math.exp(-(self.a1[i,:] - X[i])**2 / (2*self.b1[i,:]**2))
        return tf.math.multiply(tf.math.reduce_sum(h1 * self.y1, axis=1) / tf.math.reduce_sum(h1, axis=1), self.w1)
    

    def df(self, x):
        with tf.GradientTape(persistent=True) as t:
            t.watch(self.a1)
            F = self.first_layer(x)
        df_da1 = t.gradient(F, self.a1)
        return df_da1


x = [1.0,4.0]
y = []
z = 3.
#func = f(x, y, z)
#print(func.df())

X = [0.5, 0.3]
t = [0.2, 0.3]
test = DeepSIRMs(2,2)
dE_da1 = test.first_layer(X)



