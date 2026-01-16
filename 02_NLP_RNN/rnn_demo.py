import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


data=[
    "rex", "steg", "raptor", "bronto", "trike", "spino", "dino",
    "tyrann", "saur", "anka", "ptero", "mosa", "plesi"
]


chars=sorted(list(set(''.join(data))))+['.']
data_size=len(data)
vocab_size=len(chars)


print(f"数据量：{data_size}个名字")
print(f"字典大小：{vocab_size}个字符")
print(f"字典内容：{chars}")

char_to_ix={ch : ix for ix,ch in enumerate(chars)}
ix_to_char={ix:ch for ix,ch in enumerate(chars)}


class MyRNN(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(MyRNN,self).__init__()

        self.hidden_size=hidden_size


        self.i2h=nn.Linear(input_size,hidden_size)
        self.h2h=nn.Linear(hidden_size,hidden_size)
        self.h2o=nn.Linear(hidden_size,output_size)
        self.tanh=nn.Tanh()

    def forward(self,input_tensor,hidden_tensor):
        """
        前向传播：只负责处理一个时间步 (One Time Step)
        input_tensor: 当前输入的字符向量 [1, vocab_size]
        hidden_tensor: 上一步传下来的记忆 [1, hidden_size]
        """

        combined=self.i2h(input_tensor)+self.h2h(hidden_tensor)
        new_hidden=self.tanh(combined)
        output=self.h2o(new_hidden)

        return output,new_hidden
    

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)

def idx_to_tensor(index,size):
    """
    把整数索引变成 One-hot 向量
    例如: index=2, size=5 -> [0, 0, 1, 0, 0]
    """
    tensor=F.one_hot(torch.tensor([index]),num_classes=size).float()
    return tensor

n_hidden=128
learning_rate=0.005
epochs=200

rnn=MyRNN(vocab_size,n_hidden,vocab_size)

optimizer=optim.SGD(rnn.parameters(),lr=learning_rate)
criterion=nn.CrossEntropyLoss()

print("\n开始训练。。。")
for epoch in range(epochs):
    word=random.choice(data)
    input_seq='.'+word
    target_seq=word+'.'
    hidden=rnn.initHidden()
    rnn.zero_grad()
    loss=0.0

    for i in range(len(input_seq)):
        char_idx=char_to_ix[input_seq[i]]
        input_tensor=idx_to_tensor(char_idx,vocab_size)
        target_idx=torch.tensor([char_to_ix[target_seq[i]]])
        output,hidden=rnn(input_tensor,hidden)
        loss+=criterion(output,target_idx)

    loss.backward()
    optimizer.step()

    if(epoch+1)%500==0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")


print("\n训练完成：")

def evaluate(start_char='.',temperature=1.0):
    """让模型生成一个名字"""
    with torch.no_grad():
        input_tensor=idx_to_tensor(char_to_ix[start_char],vocab_size)
        hidden=rnn.initHidden()

        output_name=""
        

        for i in range(20):
            output,hidden=rnn(input_tensor,hidden)

            output_logits=output/temperature
            probs=F.softmax(output_logits,dim=1)
            top_i=torch.multinomial(probs,1).item()
            char=ix_to_char[top_i]

            if char=='.':
                break
            else:
                output_name+=char
            

            input_tensor=idx_to_tensor(top_i,vocab_size)

        return output_name


print("--- 🧊 冷静模式 (T=0.5) ---")
for _ in range(3): print(evaluate(temperature=0.5))

print("\n--- 🔥 疯狂模式 (T=1.5) ---")
for _ in range(3): print(evaluate(temperature=1.5))
for i in range(5):
    print(f"生成的名字{i+1}:{evaluate()}")