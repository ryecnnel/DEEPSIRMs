import numpy as np
from DeepSIRMs import DeepSIRMs



X = np.array([0.25, 0.5, 0.75])
p = 3               # 分割数

# 各メンバーシップ関数の中心初期値 
# 行成分は各入力xi
# 列成分は分割数p
a1 = np.array([[0.0, 0.5, 1.0],[0.0, 0.5, 1.0],[0.0, 0.5, 1.0]]) 
# 各メンバーシップ関数の幅初期値
b1 = np.array([[1.0, 0.5, 1.0],[1.0, 0.5, 1.0],[1.0, 0.5, 1.0]])

# ルール後件部初期値
# 行成分は入力xiに対するルール群内後件部
# 列成分は分割数p
y1 = np.array([[0.1, 0.1, 0.1],[0.2, 0.2, 0.2],[0.3, 0.3, 0.3]])


# ルール群重み初期値
# 各ルール群数(=入力数)に対応
w1 = np.array([0.6, 0.2, 0.3])

aaa = DeepSIRMs(X, p, a1, b1, y1, w1)
Y1 = aaa.first_layer()
Y2 = aaa.second_layer(Y1)
print(Y1, Y2)