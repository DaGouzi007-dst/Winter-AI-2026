import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, embed_size,head_size):
        super().__init__()
        self.embed_size=embed_size
        self.head_size=head_size
        self.to_q=nn.Linear(embed_size,head_size,bias=False)
        self.to_k=nn.Linear(embed_size,head_size,bias=False)
        self.to_v=nn.Linear(embed_size,head_size,bias=False)
    def forward(self,x):
        B,T,C=x.shape
        q=self.to_q(x)
        k=self.to_k(x)
        v=self.to_v(x)

        scores=q@k.transpose(-2,-1)*(1.0/math.sqrt(self.head_size))
        tril=torch.tril(torch.ones(T,T,device=x.device))
        scores=scores.masked_fill(tril==0,float('-inf'))
        weights=F.softmax(scores,dim=-1)
        out=weights@v
        return out,weights

if __name__=="__main__":
    torch.manual_seed(42)
    x=torch.randn(1,6,32)
    attention=SelfAttention(embed_size=32,head_size=16)
    output,att_weights=attention(x)
    print(f"输入形状: {x.shape}  (1句话, 6个字, 32维)")
    print(f"输出形状: {output.shape} (1句话, 6个字, 16维)")
    print("-" * 30)
    print("注意力权重矩阵 (Weights) - 第1句话:")
    print(att_weights[0].detach().numpy().round(2))

    print("_"*30)
    row_sum=att_weights[0].sum(dim=-1).detach().numpy()
    print("每一行的概率总和 (应该都等于 1):")
    print(row_sum)