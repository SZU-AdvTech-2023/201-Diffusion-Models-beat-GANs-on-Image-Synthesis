
import numpy as np

# 读取保存样本的 .npz 文件
samples_data = np.load('samples_4x64x64x3.npz')
unsamples_data=np.load('samples_16x256x256x3.npz')
unsamples150_data=np.load('samples_4x256x256x3.npz')
unsamples300_data=np.load('samples_4x256x256x3_300.npz')
unsamples1000_data=np.load('samples_4x256x256x3_1000.npz')
from pytorch_fid import fid_score
import numpy as np


arr_0 = samples_data['arr_0']
arr_1 = samples_data['arr_1']
arr_un_0 =unsamples_data['arr_0']
arr_un150_0 =unsamples150_data['arr_0']
arr_un300_0 =unsamples300_data['arr_0']
arr_un1000_0 =unsamples1000_data['arr_0']
print(samples_data.files)
print(unsamples_data.files)
import matplotlib.pyplot as plt
print(arr_0.shape)
print(arr_1.shape)
plt.subplot(4, 5, 1)

plt.imshow(arr_0[0])  # 显示 arr_0 中的第一个图像
plt.title('Image sampled')
plt.axis('off')

plt.subplot(4, 5, 2)

plt.imshow(arr_un_0[0])  # 显示 arr_1 中的第一个图像
plt.title('Image unsampled')
plt.axis('off')

plt.subplot(4, 5, 3)

plt.imshow(arr_un150_0[0])  # 显示 arr_1 中的第一个图像
plt.title('Image unsampled 150')
plt.axis('off')
plt.subplot(4, 5, 4)
plt.imshow(arr_un300_0[0])  # 显示 arr_1 中的第一个图像
plt.title('Image unsampled 300')
plt.axis('off')
plt.subplot(4, 5,5)
plt.imshow(arr_un1000_0[0])  # 显示 arr_1 中的第一个图像
plt.title('Image unsampled 1000')
plt.axis('off')

plt.subplot(4, 5, 6)

plt.imshow(arr_0[1])  # 显示 arr_0 中的第一个图像

plt.axis('off')

plt.subplot(4, 5, 7)

plt.imshow(arr_un_0[1])  # 显示 arr_1 中的第一个图像

plt.axis('off')

plt.subplot(4, 5, 8)

plt.imshow(arr_un150_0[1])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5, 9)
plt.imshow(arr_un300_0[1])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5,10)
plt.imshow(arr_un1000_0[1])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5, 11)

plt.imshow(arr_0[2])  # 显示 arr_0 中的第一个图像

plt.axis('off')

plt.subplot(4, 5, 12)

plt.imshow(arr_un_0[2])  # 显示 arr_1 中的第一个图像

plt.axis('off')

plt.subplot(4, 5, 13)

plt.imshow(arr_un150_0[2])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5, 14)
plt.imshow(arr_un300_0[2])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5,15)
plt.imshow(arr_un1000_0[2])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5, 16)

plt.imshow(arr_0[3])  # 显示 arr_0 中的第一个图像

plt.axis('off')

plt.subplot(4, 5, 17)

plt.imshow(arr_un_0[3])  # 显示 arr_1 中的第一个图像

plt.axis('off')

plt.subplot(4, 5, 18)

plt.imshow(arr_un150_0[3])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5, 19)
plt.imshow(arr_un300_0[3])  # 显示 arr_1 中的第一个图像

plt.axis('off')
plt.subplot(4, 5,20)
plt.imshow(arr_un1000_0[3])  # 显示 arr_1 中的第一个图像

plt.axis('off')
# 调整布局，避免重叠
plt.tight_layout()

plt.show()
fid_value = fid_score.calculate_fid_given_paths(['samples_4x64x64x3.npz'], ['samples_4x256x256x3.npz'], device='cuda', dims=2048)
print(f"FID between unsamples and samples: {fid_value}")