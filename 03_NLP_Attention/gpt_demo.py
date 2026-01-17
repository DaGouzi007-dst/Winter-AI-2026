import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from multi_head_demo import Block

class GPT(nn.Module):
    def __init__(self, vocab_size,embed_size,num_heads,num_layers,max_len):
        super().__init__()
        self.token_embedding=nn.Embedding(vocab_size,embed_size)
        self.position_embedding=nn.Embedding(max_len,embed_size)
        self.blocks=nn.Sequential(*[
            Block(embed_size,num_heads) for _ in range(num_layers)
        ])

        self.ln_f=nn.LayerNorm(embed_size)
        self.lm_head=nn.Linear(embed_size,vocab_size)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        tok_emb=self.token_embedding(idx)
        pos_emb=self.position_embedding(pos)
        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            criterion=nn.CrossEntropyLoss()
            loss=criterion(logits,targets)
        return logits,loss
    
    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-self.position_embedding.num_embeddings:]
            logits,_=self(idx_cond)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=-1)
        return idx
        


if __name__=="__main__":
    torch.manual_seed(1337)

    try:
        with open('input.txt','r',encoding='utf-8') as f:
            text=f.read()
        print(f"成功读取数据，文本长度：{len(text)}")

    except FileExistsError:
        print("没找到input.txt")

    chars=sorted(list(set(text)))
    vocab_size=len(chars)
    print(f"词表大小：{vocab_size}")
    print(f"包含字符：{''.join(chars[:20])}....")

    stoi={ch:i for i,ch in enumerate(chars)}
    itos={i:ch for i,ch in enumerate(chars)}
    encode=lambda s:[stoi[c] for c in s]
    decode=lambda l :[''.join([itos[i] for i in l])]

    data=torch.tensor([encode(text)],dtype=torch.long)
    model=GPT(vocab_size=vocab_size,embed_size=64,num_heads=4,num_layers=4,max_len=64)

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)
    data=data.view(-1)
    data=data.to(device)
    print(f"DEBUG: data.shape = {data.shape}")
    print(f"正在使用设备：{device}")

    optimizer=optim.AdamW(model.parameters(),lr=1e-3)
    batch_size=32
    block_size=64
    max_iters=3000

    print("开始训练莎士比亚风格。。。")

    for step in range(max_iters):
        ix=torch.randint(len(data)-block_size,(batch_size,))
        x=torch.stack([data[i:i+block_size] for i in ix])
        y=torch.stack([data[i+1:i+block_size+1] for i in ix])
        x,y=x.to(device),y.to(device)

        logits,loss=model(x,y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step%100==0:
            print(f"Step{step}:Loss={loss.item():.4f}")

    print("训练完成。。。")

    print("_"*30)
    print("AI生成的莎士比亚剧本：")
    print("_"*30)
    
    context=torch.zeros((1,1),dtype=torch.long,device=device)

    output=model.generate(context,max_new_tokens=500)

    print(decode(output[0].tolist()))









 
    # text="I love deep learning!"
    # chars=sorted(list(set(text)))
    # vocab_size=len(chars)
    # print(f"词表大小：{vocab_size}")
    # print(f"词表：{chars}")

    # stoi={ch:i for i,ch in enumerate(chars)}
    # itos={i:ch for i,ch in enumerate(chars)}
    # encode=lambda s:[stoi[c] for c in s]
    # decode=lambda l:''.join([itos[i] for i in l])

    # data=torch.tensor(encode(text),dtype=torch.long)
    # model=GPT(vocab_size,embed_size=32,num_heads=4,num_layers=2,max_len=32)

    # optimizer=optim.AdamW(model.parameters(),lr=1e-3)
    # print("开始训练。。。")
    
    # for step in range(1000):

    #     block_size=8
    #     batch_size=4

    #     ix=torch.randint(len(data)-block_size,(batch_size,))
    #     x=torch.stack([data[i:i+block_size] for i in ix])
    #     y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    #     logits,loss=model(x,y)

    #     optimizer.zero_grad(set_to_none=True)
    #     loss.backward()
    #     optimizer.step()

    #     if step%100==0:
    #         print(f"Step{step}：Loss：{loss.item():.4f}")

    # print("训练完成。。。")
    # print("_"*30)
    # context=torch.tensor([[stoi['I']]],dtype=torch.long)
    # generated_ids=model.generate(context,max_new_tokens=20)[0].tolist()
    # print("生成结果：",decode(generated_ids))























    # torch.manual_seed(42)
    # vocab_size=1000
    # embed_size=32
    # num_heads=4
    # num_layers=3
    # max_len=20

    # model=GPT(vocab_size,embed_size,num_heads,num_layers,max_len)
    # print("模型构建成功")
    # print(model)
    # input_idx=torch.randint(0,vocab_size,(1,6))
    # logits=model(input_idx)
    # context=torch.zeros((1,1),dtype=torch.long)
    # print("-" * 30)
    # print("🤖 AI 正在尝试瞎编生成...")
    # generated_ids = model.generate(context, max_new_tokens=10)
    
    # print(f"生成的序列 ID: {generated_ids.tolist()[0]}")
    # print("-" * 30)
    # print("✅ 成功！虽然它现在只会输出乱码，但它已经学会了‘接龙’！")

    # print("-" * 30)
    # print(f"输入形状: {input_idx.shape} (1句话, 6个字)")
    # print(f"输出形状: {logits.shape} (1句话, 6个字, 1000个预测概率)")

    # if logits.shape == (1, 6, vocab_size):
    #     print("✅ 测试通过！这就是一个未经训练的 Mini-GPT！")