#本地推理脚本
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel#PeftModel是把LoRA权重挂载到基座模型上
from config import MODEL_NAME,LORA_SAVE_PATH
def qwen_infer(system_prompt,user_query,temp=0.7,top_p=0.9):#定义一个Qwen+LoRA微调模型的推理接口，system_prompt是系统角色提示词，user_query是用户输入问题，temp是温度系数，越高回答越天马行空，越低越严谨，top_p是核采样，累积概率前90%token 都参与，过滤低概率生僻词
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
    base=AutoModelForCausalLM.from_pretrained(MODEL_NAME,torch_dtype=torch.bfloat16,device_map="auto",trust_remote_code=True)
    model=PeftModel.from_pretrained(base,LORA_SAVE_PATH)
    #Qwen对话模板
    prompt=f"<|im_start|>systeem\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<im_start>assistant\n"#手动拼接Qwen对话prompt
    inputs=tokenizer(prompt,return_tensor="pt").to("cuda")#文本转模型输入张量迁移到gpu设备
    outputs=model.generate(
        **inputs,#解包input_ids,attention_mask传入生成函数
        max_new_tokens=512,#限制模型最多新生成512个字
        temperature=temp,#温度系数
        top_p=top_p,#核采样
        do_sample=True#开启随机采样，搭配temp和top_p生效，若关闭则贪心搜索，每次选概率最高词，回答固定无变化
    )#模型生成推理核心generate
    res=tokenizer.decode((outputs[0][len(inputs["inputs_ids"][0]):]),#outputs[0]是取第一条推理结果，[len....]是取inputs解包的inputs_ids的第一项切片，即切掉前面输入的prompt内容，只保留模型新生成的文字
                         skip_special_tokens=True)#decode是解码输出，skip_special_tokens是去除<|im_start|>之类特殊标记，返回干净文本
    return res
if __name__ == "__main__":
    sys_prompt="你是专业内科问诊助手"
    question="经常熬夜胸闷气短是什么原因"
    ans=qwen_infer(sys_prompt,question)
    print("模型回答：",ans)

