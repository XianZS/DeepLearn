import torch

a = torch.cuda.is_available()  # type:ignore
print(a)
