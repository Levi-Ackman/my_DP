import torch
import math
import torch.nn as nn
max_len,embed_dim=100,128
positon_code=torch.zeros(max_len,embed_dim)
p1=torch.tensor([[1,2,3,4,5,6,7,8,9,10]])

x=p1[:,0::2]
y=p1[:,1::2]
print(x)
print(f'x_shape{x.shape}')
print(y)
print(f'y_shape{y.shape}')

p2=torch.arange(0,max_len).unsqueeze(1)
p3=torch.arange(0,embed_dim,2)
print(p3)
t=p2*torch.exp(p3*(-math.log(10000.0)/embed_dim))
print(f'dim of p2: {p2.shape}  dim of p3: {p3.shape}  dim of t: {t.shape}')

psc=torch.zeros(max_len,embed_dim)
psc[:,1::2]=torch.sin(t)
psc[:,0::2]=torch.cos(t)

print(f'position_embedding:  {psc}')
print(psc.shape)

ori=torch.zeros(10,dtype=int)
print(f'original : {ori}, after torch.exp(): {torch.exp(ori)}, after torch.cos(): {torch.cos(ori)} ')
mask=torch.IntTensor([0,1,0,1,1,1,1,0,1,1])
ori_after_masked=ori.masked_fill(mask==0,-1e9)
print('/n'+f'original tensor: {ori}, masked tensor: {ori_after_masked}')


print('ji',str(1))

print('--'*20)

x=torch.rand(1,1,4)
print(x.shape)
x=x.expand(4,-1,-1)
print(x.shape)

