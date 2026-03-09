import torch

flag = torch.cuda.is_available()
print(flag)

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())

cuda_version = torch.version.cuda
print(f"CUDA Version: {cuda_version}")

"""
D:\code\py_item\DeepLearn\One_Base_Learning\pytorch_framework>python select_gpu.py
True
cuda:0
NVIDIA GeForce RTX 3050 Laptop GPU
tensor([[0.7245, 0.5391, 0.9066],
        [0.3770, 0.1952, 0.8257],
        [0.5377, 0.7795, 0.7644]], device='cuda:0')
CUDA Version: 11.3
"""
