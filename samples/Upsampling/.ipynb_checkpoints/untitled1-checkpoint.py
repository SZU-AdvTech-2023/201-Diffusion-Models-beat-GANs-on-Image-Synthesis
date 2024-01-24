from pytorch_fid import fid_score
import numpy as np# 从两个 npz 文件中加载数据
low_res_samples = np.load('samples_4x64x64x3.npz')['arr_0']
high_res_samples = np.load('samples_4x256x256x3.npz')['arr_0']

# 确保两个数据集的形状相同
assert low_res_samples.shape == high_res_samples.shape, "Dataset sizes do not match!"
# 使用 FID 计算
fid_value = fid_score.calculate_fid(low_res_samples, high_res_samples, device='cuda', dims=2048)
print(f"FID Value: {fid_value}")
f"FID: {fid_value}")