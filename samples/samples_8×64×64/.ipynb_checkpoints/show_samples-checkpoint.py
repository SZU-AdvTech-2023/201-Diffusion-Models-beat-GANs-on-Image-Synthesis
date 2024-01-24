import numpy as np
import matplotlib.pyplot as plt
# 从npz文件加载数据
data = np.load('samples_8x64x64x3.npz')

# 打印文件中的数组名字
print(data.files)

# 获取数组
arr_0 = data['arr_0']

plt.subplot(2, 4, 1)
plt.imshow(arr_0[0])  # 显示 arr_0 中的第一个图像

plt.axis('off')

plt.subplot(2, 4, 2)

plt.imshow(arr_0[1])  # 显示 arr_1 中的第一个图像

plt.axis('off')

plt.subplot(2, 4, 3)

plt.imshow(arr_0[2])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(2, 4, 4)
plt.imshow(arr_0[3])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(2, 4,5)
plt.imshow(arr_0[4])  # 显示 arr_1 中的第一个图像

plt.axis('off')

plt.subplot(2, 4, 6)

plt.imshow(arr_0[5])  # 显示 arr_0 中的第一个图像

plt.axis('off')

plt.subplot(2, 4, 7)

plt.imshow(arr_0[6])  # 显示 arr_1 中的第一个图像

plt.axis('off')

plt.subplot(2, 4, 8)

plt.imshow(arr_0[7])  # 显示 arr_1 中的第一个图像

plt.axis('off')

plt.show()