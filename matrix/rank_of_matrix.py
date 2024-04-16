import numpy as np

a = np.array([[1, 0, 0],
             [6, 5, 0],
             [4, 1, 15],
             [52, 8, 5]])
b = np.array([[74, 45, 5],
             [74, 45, 5],
             [74, 45, 5],
             [74, 45, 5]])
# 将a b 矩阵横向拼接为c
c = np.append(b, a, axis=1)
# 打印矩阵的秩
print(np.linalg.matrix_rank(a))   # 3
print(np.linalg.matrix_rank(b))   # 1
print(c)
"""
[[74 45  5  1  0  0]
 [74 45  5  6  5  0]
 [74 45  5  4  1 15]
 [74 45  5 52  8  5]]
"""
print(np.linalg.matrix_rank(c))   # 4