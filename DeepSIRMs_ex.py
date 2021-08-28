# 森友樹さんの修論内容のプログラミング
# p.11の例のコードになります
import numpy as np

l = 2
p = 3
X = np.array([0.25, 0.5, 0.75]) # input data x1 x2 x3

#メンバーシップ関数の定義(三角型)
# 一層目ルール部
Ax1 = np.zeros((p,p)) # membership function Aij(x)
a1 = np.array([[0.0, 0.5, 1.0],
               [0.0, 0.5, 1.0],
               [0.0, 0.5, 1.0]])
b1 = np.array([[1.0, 0.5, 1.0],
               [1.0, 0.5, 1.0],
               [1.0, 0.5, 1.0]])
y1 = np.array([[0.1, 0.1, 0.1],
               [0.2, 0.2, 0.2],
               [0.3, 0.3, 0.3]])
w1 = np.array([0.6, 0.2, 0.3])
yl = np.zeros(l)

# 一層目出力
for i in range(p):
    Ax1[i,:] = 1 - np.abs(a1[i,:]-X[i]) / b1[i,:] #grade cal    
yl[0] = np.dot(np.sum(Ax1 * y1, axis=1) / np.sum(Ax1,axis=1), w1)

# 二層目出力
a2 = np.append(a1, [[0.0, 0.5, 1.0]], axis=0)
b2 = np.append(b1, [[1.0, 0.5, 1.0]], axis=0)
y2 = np.append(y1, [[0.4, 0.4, 0.4]], axis=0)
w2 = np.append(w1, 0.7)
Ax2_add = 1 - np.abs(a2[p, :]-yl[0]) / b2[p,:]
Ax2 = np.append(Ax1, [Ax2_add], axis=0)

# 二層目出力 
yl[1] = np.dot(np.sum(Ax2 * y2, axis=1) / np.sum(Ax2, axis=1), w2)
print(yl)


