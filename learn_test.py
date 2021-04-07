import numpy as np
from keras.utils import to_categorical
import os
log_path = os.getcwd() + "\\log"
print(log_path)
# # One-hot 编码
# a = [i for i in range(10)]
# print(a)
# a_np = np.array(a)
# a_one_hot = to_categorical(a_np, 0)
# print(a_one_hot)