import numpy as np

def random_sample(dataset, num_samples):
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return dataset[indices]

# 从两个数据集中随机采样相同数量的样本
num_samples = 100  # 你可以根据需要调整采样的样本数量
samples_data = np.load('samples_4x64x64x3.npz')['arr_0']
unsamples_data = np.load('samples_4x256x256x3.npz')['arr_0']

random_samples = random_sample(samples_data, num_samples)
random_unsamples = random_sample(unsamples_data, num_samples)
from pytorch_fid import fid_score

def calculate_fid(sample1, sample2):
    fid_value = fid_score.calculate_fid(sample1, sample2, device='cuda', dims=2048)
    return fid_value

fid_value = calculate_fid(random_samples, random_unsamples)
print(f"FID: {fid_value}")
