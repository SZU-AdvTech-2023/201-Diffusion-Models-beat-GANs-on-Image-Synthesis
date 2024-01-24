import numpy as np

# 从npz文件加载数据
data = np.load('samples_8x64x64x3.npz')

# 打印文件中的数组名字
print(data.files)

# 获取数组
array1 = data['arr_0']
array2 = data['arr_1']
print(array1.shape)
print(array2)

# 关闭文件
data.close()