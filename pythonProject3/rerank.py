import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer#导入自动分词器和预训练文本分类模型加载工具
from config import RERANK_MODEL_NAME,DEVICE
class Reranker:#定义重排工具类，封装所有重排能力，RAG架构中，该类专门封装精排逻辑
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)#加载对应预训练分词器，self实例全局变量，整个类都能用。from_pretrained():HF核心加载函数，自动下载读取模型配套的vocab词表，分词规则。
        self.model=AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL_NAME).to(DEVICE)#加载带分类头的交叉编码器模型，并迁移指定GPU/CPU设备
        self.model.eval()#设置模式为评估模式，train是训练模式，评估模式做推理，搭配torch.no_grad()防止梯度计算，省内存
    def rerank(self,query:str,chunk_list:list,top_n:int):#定义重排核心业务函数。query：用户输入检索问句，chun_list：粗召回后的文档片段列表，top_n:重排后保留前N个最匹配文档
        pairs=[]#空列表，用来存放【查询-文档】配对
        for chunk in chunk_list:#遍历所有召回片段
            pairs.append([query,chunk["text"]])
        inputs=self.tokenizer(pairs,padding=True,truncation=True,max_length=512,return_tensors="pt").to(DEVICE)#padding自动给短文本补0，使每一段序列长度统一。truncation是超文本自动截断。然后输出张量PyTorch
        with torch.no_grad():#临时关闭梯度计算
            scores=self.model(**inputs).logits.squeeze()#解包inputs字典（input_ids,attention_mask等）送入交叉编码器前向推理，.logits模型分类头输出，代表匹配原始分值。.squeeze()压缩多余维度，方便后续按索引取分
        for idx,chunk in enumerate(chunk_list):#遍历解包赋值，enumerate是内置函数，给列表遍历带上序号，方便对齐scores数组
            chunk["rerank_score"]=float(scores[idx].cpu())#scores[idx]是张量，不能直接存进JSON/接口返回，必须转为普通浮点数
        sorted_chunks=sorted(chunk_list,key=lambda x:x["rerank_score"],reverse=True)#sorted()是Python内置的排序函数，生成全新列表，不会修改原列表。key是排序依据参数，指定字段作比较。lambda是匿名函数，x代表列表里面的每一个chunk字典，然后取每条文档的匹配分数作为排序权重。reverse=True是降序
        return sorted_chunks[:top_n]#[:top_n]代表取列表前top_n个元素，分数由高到低拍好，取前N条最匹配文档返回
