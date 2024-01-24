import numpy as np
import matplotlib.pyplot as plt

def show_images_from_npz(npz_file):
    # 加载 .npz 文件
    data = np.load(npz_file)

    # 获取图像数组
    images = data['arr_0']  # 请根据实际情况修改数组名

    # 确保 images 是一个形状为 (100, 64, 64, 3) 的数组
    if images.shape == (100, 64, 64, 3):
        # 创建一个 10x10 的子图网格，每个子图显示一张图像
        fig, axes = plt.subplots(10, 10, figsize=(10, 10))
        for i in range(10):
            for j in range(10):
                axes[i, j].imshow(images[i * 10 + j])
                axes[i, j].axis('off')  # 关闭坐标轴

        plt.show()
    else:
        print("图像数组的形状不正确，请检查数据。")

# 替换 'your_file.npz' 为你的文件路径
show_images_from_npz('samples_100x64x64x3.npz')
