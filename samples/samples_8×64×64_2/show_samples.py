
import numpy as np

# 读取保存样本的 .npz 文件
samples_data = np.load('samples_8x64x64x3.npz')
arr_0 = samples_data['arr_0']
arr_1 = samples_data['arr_1']

print(samples_data.files)
import matplotlib.pyplot as plt
print(arr_0.shape)
print(arr_1.shape)
plt.subplot(1, 8, 1)

plt.imshow(arr_0[0])  # 显示 arr_0 中的第一个图像
plt.title('Image from arr_0')
plt.axis('off')

plt.subplot(1, 8, 2)

plt.imshow(arr_0[1])  # 显示 arr_1 中的第一个图像
plt.title('Image from arr_1')
plt.axis('off')
plt.subplot(1, 8, 3)
plt.imshow(arr_0[2])  # 显示 arr_1 中的第一个图像
plt.title('Image from arr_1')
plt.axis('off')
plt.subplot(1, 8, 4)
plt.imshow(arr_0[3])  # 显示 arr_1 中的第一个图像
plt.title('Image from arr_1')
plt.axis('off')
plt.subplot(1, 8, 5)
plt.imshow(arr_0[4])  # 显示 arr_1 中的第一个图像
plt.title('Image from arr_1')
plt.axis('off')
plt.subplot(1, 8, 6)
plt.imshow(arr_0[5])  # 显示 arr_1 中的第一个图像
plt.title('Image from arr_1')
plt.axis('off')
plt.subplot(1, 8, 7)
plt.imshow(arr_0[6])  # 显示 arr_1 中的第一个图像
plt.title('Image from arr_1')
plt.axis('off')
plt.subplot(1, 8, 8)
plt.imshow(arr_0[7])  # 显示 arr_1 中的第一个图像
plt.title('Image from arr_1')
plt.axis('off')


# 调整布局，避免重叠
plt.tight_layout()

plt.show()