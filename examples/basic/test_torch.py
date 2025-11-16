import torch 


a = torch.rand(2)

print(torch.log(a))
print(torch.std(a))
print(torch.tanh(a))
print(torch.relu(a))


# print(torch.random.randn(3))

mod = torch
mod.random.uniform(shape=(1, 2), minval=-2, maxval=2, dtype=torch.float32)