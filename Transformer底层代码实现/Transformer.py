import torch   #torch基础张量库
import torch.nn as nn   #神经网络层容器（Linear，Conv，LayerNorm等）
import torch.nn.functional as F   #无参数的静态函数库
#缩放点积注意力（Transformer的核心，是多头注意力的底层基础，是原生Pytorch的实现）
def scaled_dot_product_attention(q,k,v,mask=None):#q,k,v的核心定义：q：当前token要找什么，要匹配的查询；k：提供什么，全局所有token的特征键；v：内容，最终加权提取的内容值。它们的形状是[batch,head_num,seq_len,d_k]
    d_k=q.size(-1)
    attn=torch.matmul(q,k.transpose(-2,-1))/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))#q@k.T(matmul矩阵乘法），计算每个Q和所有K的相似度，输出形状[B,H,lq,lk]。为什么要除以根号dk，因为dk很大时，会影响softmax时训练不稳定
    if mask is not None:
        attn=attn.masked_fill_(mask==0,-1e9)#原地填充操作，使mask为无效位置，不参与加权求和
        attn=F.softmax(attn,dim=-1)#对k序列维度做归一化，每行q所有权重之和=1
        out=torch.matmul(attn,v)
        return out
#多头注意力
class MultiHeadAttention(nn.Module):#自定义神经网络层类，继承了Pytorch内置网络层类
    def __init__(self,d_model,n_heads):#类的初始化，传入两个超参
        super().__init__()#调用父类
        assert d_model%n_heads==0#断言校验，保证总维度均分到每个注意力头
        self.d_model=d_model
        self.n_heads=n_heads
        self.d_k=d_model/n_heads#单头维度
        self.wq=nn.Linear(d_model,d_model)#线性层投影，把输入特征映射成qkv
        self.wk=nn.Linear(d_model,d_model)
        self.wv=nn.Linear(d_model,d_model)
        self.wo=nn.Linear(d_model,d_model)#多头输出融合层，用来整合
        self.register_buffer('sqrt_dk',torch.sqrt(torch.tensor(self.d_k,dtype=torch.float32)))#预存根号d_k
    def split_heads(self,x):#定义单个头函数，并传入形状参数x
        B,L,D=x.shape
        x=x.view(B,L,self.n_heads,self.d_k)#x.view重塑维度
        x=x.transpose(1,2).contiguous()
        return x
    def forward(self,q,k,v,mask=None):#定义前向传播函数
        B=q.size(0)#获取批次大小，后续复原张量要用
        q=self.split_heads(self.wq(q))#调用split_heads函数拆分多头
        k=self.split_heads(self.wk(k))
        v=self.split_heads(self.wv(v))
        attn_scores=torch.matmul(q,k.transpose(-2,-1))/self.sqrt_dk
        if mask is not None:
            attn_scores=attn_scores.masked_fill_(mask==0,-1e9)
            attn_scores=F.softmax(attn_scores,dim=-1)
            out=torch.matmul(attn_scores,v)
            out=out.transpose(1,2).contiguous()#多头拼接复原，.contiguous保证张量内存连续
            out=out.view(B,-1,self.d_model)
            return self.wo(out)#线性层融合多头特征，返回输出
#层归一化（这是对单个句子或单条样本的内部特征维度做归一化，作用是稳定训练分布，缓解梯度消失，加速模型收敛。Transformer编码器和解码器每一层会前后使用一次）
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-6):#传入模型特征维度和极小常数，这个常数是防止分母为0
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(d_model))#nn.Parameter代表可训练权重，优化器会更新。*alpha用来恢复特征的幅值，避免把特征压缩到过小区间
        self.beta=nn.Parameter(torch.zeros(d_model))#beta是偏移可学习参数，也是让归一化不破坏模型表达特征能力
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)#取特征维度做均值，保留计算后的维度。形状为[B,L,1]，作用是保证后面和原张量x做广播运算维度匹配
        std=x.std(-1,keepdim=True)#同样是[B,L,1]
        return self.alpha*(x-mean)/(std+self.eps)+self.beta
#前馈网络（多头注意力负责捕捉token之间的全局依赖关系，但是前馈网络可以对每一个token单独做非线性特征交换）
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff=2048,dropout=0.1):#d_ff是隐藏层维度，默认是2048，将模型特征维度放大四倍，提取更多特征细节，相当于图片放大看。dropout是防止模型过拟合
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff)
        self.linear2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)#防过拟合
        self.relu=nn.ReLU()#激活函数ReLU引入非线性，不引入则两层Linear等价于一层线性交换，大大降低模型拟合度。隐藏层基本用Relu激活函数，现在也可以用GELU代替Relu，前者效果更好。
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)#还原维度，方便后续做残差连接，解决深层网络梯度消失问题
        return x
#位置编码（Transformer只有矩阵乘法注意力机制，没有循环卷积结构，无法感知句子先后顺序。位置编码就是给每个位置生成一组独一无二的正余弦特征向量，叠加到词嵌入上，让模型区分token的先后位置）
class PositionEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000,dropout=0.1):#max_len是预先缓存的最大序列长度。dropout防过拟合，弱化位置与词嵌入的耦合
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model)#位置编码矩阵，形状为[max_len,d_model]
        pos=torch.arange(0,max_len,dtype=torch.float32).unsqueeze(1)#生成0-max_ken-1连续位置数字，表示句子中每个token的绝对位置，[max_len,1]
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-torch.log(torch.tensor(10000.0))/d_model))#正余弦编码频率系数:维度i越大，div_term越小，则正余弦周期越长。低维度捕捉局部位置差异，高维度捕捉远距离位置依赖
        pe[:,0::2]=torch.sin(pos*div_term)#对所有偶数维度填充正弦值
        pe[:,1::2]=torch.cos(pos*div_term)#对所有奇数维度填充余弦值
        self.register_buffer('pe',pe)#注册是为模型缓冲区，不是可训练参数，不会参与梯度更新，也不会被优化器修改
    def forward(self,x):
        x=x+self.pe[:x.size(1)]#广播相加，词向量+位置向量（一个负责语义，一个负责时序顺序），二者融合送入后续注意力层。它是叠加不是拼接，维度是不变的
        return self.dropout(x)
#Encoder层(单层编码器，使用自注意力即QKV全部来自输入本身，这段采用Post-LN结构）固定结构：自注意力——残差——归一化——前馈——残差——归一化
class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,n_heads,dropout=0.1):
        super().__init__()
        self.attn=MultiHeadAttention(d_model,n_heads)#入参都是它本来有的参数
        self.ffn=FeedForward(d_model,d_ff,dropout)
        self.norm1=LayerNorm(d_model)
        self.norm2=LayerNorm(d_model)
        self.drop1=nn.Dropout(dropout)
        self.drop2=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        attn_out=self.attn(x,x,x,mask)#x融合了词嵌入加位置编码的特征[b,l,d_model]，编码器是自注意力，所以QKV全部传入同一个x
        x=self.norm1(x+self.drop1(attn_out))
        ffn_out=self.ffn(x)
        x=self.norm2(x+self.drop2(ffn_out))
        return x
#Decoder层（单层解码器:三层计算分支，掩码自注意力，交叉注意力，前馈网络）
class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,n_heads,dropout=0.1):
        super().__init__()
        self.masked_attn=MultiHeadAttention(d_model,n_heads)
        self.cross_attn=MultiHeadAttention(d_model,n_heads)
        self.ffn=FeedForward(d_model,d_ff,dropout)
        self.norm1=LayerNorm(d_model)
        self.norm2=LayerNorm(d_model)
        self.norm3=LayerNorm(d_model)
        self.drop1=nn.Dropout(dropout)
        self.drop2=nn.Dropout(dropout)
        self.drop3=nn.Dropout(dropout)
    def forward(self,x,enc_out,tgt_mask=None,src_mask=None):#x词嵌入加位置编码特征，enc_out是编码器全部层输出，tgt_mask目标序列掩码，双重作用：下三角：屏蔽未来token。padding：屏蔽目标句末填充符。src_mask源序列padding掩码，交叉注意力中屏蔽源句子无效填充位置
        attn1=self.masked_attn(x,x,x,tgt_mask)
        x=self.norm1(x+self.drop1(attn1))
        attn2=self.cross_attn(x,enc_out,enc_out,src_mask)
        x=self.norm2(x+self.drop2(attn2))
        ffn_out=self.ffn(x)
        x=self.norm3(x+self.drop3(ffn_out))
        return x
#掩码生成工具
def create_pad_mask(src,pad_idx=0):#填充掩码，生成Padding掩码，屏蔽句子末尾填充的占位符token
    return (src!=pad_idx).unsqueeze(1).unsqueeze(1)#布尔判断，token不等于填充符返回True（有效位置）。为什么要升维度？因为多头注意力QKV是四维，mask要和attn_score广播匹配。[B,1,1,lk]
def create_causal_mask(seq_len):#因果掩码，解码器掩码自注意力专用，防止看未来token导致泄密
    mask=torch.tril(torch.ones(seq_len,seq_len))#提取下三角，构造全是1的下三角，上三角为0
    return mask.unsqueeze(0).unsqueeze(0)#[1,1,lq,lk]
#Transformer最终版
class Transformer(nn.Module):#主模型类，分为编码端和解码端两大模块。（整体执行流程：源文本编码提取全局上下文 → 目标文本逐词解码生成译文，内部整合词嵌入、位置编码、多层编码器、多层解码器、最终分类输出层，同时包含掩码生成、嵌入缩放等细节优化。）
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model=512,n_heads=8,n_layers=6,d_ff=2048,dropout=0.1,pad_idx=0):
        #保存全局超参
        self.d_model=d_model
        self.pad_idx=pad_idx
        #词嵌入层
        self.src_emb=nn.Embedding(src_vocab_size,d_model)
        self.tgt_emb=nn.Embedding(tgt_vocab_size,d_model)
        #实例位置编码
        self.pe=PositionEncoding(d_model,dropout=dropout)
        self.register_buffer('sqrt_dmodel',torch.tensor(d_model,dtype=torch.float32).sqrt())
        self.enc_layers=nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff,dropout)for _ in range(n_layers)])#编码器堆栈，叠堆6层。nn.ModuleList专门存放多层网络层，自动注册参数。
        self.dec_layers=nn.ModuleList([DecoderLayer(d_model,n_heads,d_ff,dropout)for _ in range(n_layers)])#解码器堆栈，叠堆6层
        self.fc=nn.Linear(d_model,tgt_vocab_size)#最终输出线性层，将解码器输出d_model维特征映射到目标词表维度
    def forward(self,src,tgt):
        #生成掩码
        src_mask=create_pad_mask(src,self.pad_idx)#源文本掩码，屏蔽padding字符，用于编码器自注意力和解码器交叉注意力使用
        tgt_len=tgt.size(1)#tgt结构：tgt[batch,tgt_len]
        tgt_mask=create_causal_mask(tgt_len)#下三角因果掩码，防止看到未来token，用于解码器掩码自注意力使用
        #Encoder前向 嵌入缩放+位置编码(编码端）
        src_emb=self.src_emb(src)*self.sqrt_dmodel#源文本token经过嵌入层得到向量*根号dmodel。乘根号dmodel放大数值和注意力内积数值匹配。
        src_emb=self.pe(src_emb)#再送入位置编码，叠加时序信息和dropout正则
        enc_out=src_emb
        for layer in self.enc_layers:#循环遍历全部编码器层，特征逐层更新，得到最终的enc_out
            enc_out=layer(enc_out,src_mask)
        #Decoder前向
        tgt_emb=self.tgt_emb(tgt)*self.sqrt_dmodel
        tgt_emb=self.pe(tgt_emb)
        dec_out=tgt_emb
        for layer in self.dec_layers:
            dec_out=layer(dec_out,enc_out,tgt_mask,src_mask)
            return self.fc(dec_out)
#测试运行
if __name__ == "__main__":#模型独立测试入口，这是Python标准程序入口判断语句
    model=Transformer(src_vocab_size=1000,tgt_vocab_size=1000)
    src=torch.randint(0,1000,(2,15))#随机生成0-999之间的整数。形状是（2，15），一批两条句子，每条源句子长度15.[batch,len]
    tgt=torch.randint(0,1000,(2,10))
    out=model(src,tgt)
    print("输出shape：",out.shape)
































































































