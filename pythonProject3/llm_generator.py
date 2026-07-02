import  torch
from transformers import AutoModelForCausalLM,AutoTokenizer#导入因果语言模型自动加载器和自动分词器
from config import LLM_MODEL_NAME,DEVICE#LLM是大语言模型
class LLMGenerate:#定义一个语言大模型推理的类
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained(LLM_MODEL_NAME,trust_remote_code=True)#trust是信任模型仓库里自定义的分词代码
        self.model=AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=DEVICE,
            trust_remote_code=True
        )#权重使用半精度浮点存储，自动把模型权重分配指定硬件设备
    def build_prompt(self,query:str,context:list)->str:#用户初始提问，字符串类型；RAG检索出来的知识库片段列表，列表每个元素都是字典，->str代表函数最终返回拼接好的完整提示词字符串
        context_text="\n".join([f"参考片段：{c['text']}" for c in context])#遍历检索到的所有知识库片段，"\n".join把多条知识库片段用换行符拼接成一整段字符串,方便塞进prompt模板
        prompt=f"""
        你是企业知识库问答助手，严格遵守以下规则：
1. 仅允许使用【参考片段】中的内容回答问题，绝对不能编造不存在的信息；
2. 如果参考片段无相关内容，直接回复：【知识库未查询到相关信息】；
3. 回答时标注对应参考片段来源；
4. 回答简洁专业，禁止输出无关内容。
【参考片段】
{context_text}
【用户问题】
{query}
回答：
"""
        return prompt#第一部分设定模型身份，定义它是企业知识库专属问答助手，限定业务场景；
#第二部分四条核心规则是 RAG 项目关键，专门用来抑制大模型幻觉：强制只能依据参考资料作答、无资料统一固定回复、要求标注片段来源方便溯源、约束回答简洁专业，适配企业内部使用规范；
#第三部分填充两块动态内容：前面处理好的知识库参考片段，以及用户当前提问，最后用 “回答：” 引导模型直接输出答案，减少冗余输出。
    def generate(self,prompt:str):#定义一个推理函数，传入prompt参数
        inputs=self.tokenizer(prompt,return_tensor="pt").to(DEVICE)#输入分词器编码文本prompt转化成张量让模型接受，把编码后的张量数据迁移到GPU
        outputs=self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.3
        )#输出调用model.generate执行自回归文本生成，解包上面编码得到的inpu_ids,attention_mask。再限制模型最多生成512个字，温度系数越接近0，则输出更严谨；越大则越有创意，RAG知识库问答场景必须调低，抑制幻觉。top_p核采样：保留累加前30%的token候选，进一步减少随机
        res=self.tokenizer.decode(outputs[0],skip_special_tokens=True)#对模型输出的token序列解码，也就是数字转回文字，outputs[0]取batch第一条结果，即单条问答。然后        res=self.tokenizer.decode(outputs[0],skip_special_tokens=True)#对模型输出的token序列解码，也就是数字转回文字，outputs[0]取batch第一条结果，即单条问答。然后再自动过滤eos，pad等特殊符号，输出干净文本
        answer=res.split("回答:")[-1].strip()#再截取“回答”分割后的最后一段即答案，.strip去除收尾换行，美化输出
        return answer
