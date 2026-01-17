import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_demo import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size,num_heads):
        super().__init__()
        self.num_heads=num_heads
        self.head_size=embed_size//num_heads
        self.heads=nn.ModuleList([
            SelfAttention(embed_size,self.head_size) for _ in range(num_heads)
        ])
        self.proj=nn.Linear(embed_size,embed_size)

    def forward(self,x):
        head_outputs=[head(x)[0] for head in self.heads]
        out=torch.cat(head_outputs,dim=-1)
        out=self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(embed_size,4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size,embed_size)
        )

    def forward(self,x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_size,num_heads):
        super().__init__()
        self.sa=MultiHeadAttention(embed_size,num_heads)
        self.ffwd=FeedForward(embed_size)
        self.ln1=nn.LayerNorm(embed_size)
        self.ln2=nn.LayerNorm(embed_size)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x

if __name__=="__main__":
    torch.manual_seed(42)
    embed_size=32
    num_heads=4
    x=torch.randn(1,6,embed_size)
    block=Block(embed_size,num_heads)
    output=block(x)


    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    if x.shape == output.shape:
        print("✅ Block 构建成功！输入输出维度一致，可以无限堆叠！")
        print("🎉 恭喜！你已经亲手造出了 Transformer 的心脏！")
    else:
        print("❌ 维度不匹配，检查一下 Linear 层的输入输出大小！")