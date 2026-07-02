#训练主程序
import torch
from datasets import load_dataset#HuggingFace官方数据库
from  transformers import (AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments)#因果大模型自动加载类，自动分词器类，4/8bit量化配置类，训练超参配置类
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training#从PEFT高效参数微调库导入LoraConfig微调超参类和传入大模型，把微调适配器传入大模型
from config import *#从自定义config文件传入*（代表全部）
from trl import SFTTrainer#从trl库传入SFTT训练，支持text字段数据集，自动识别对话，原生Trainer需要手动写数据预处理等
def train_lora():#定义一个启动LoRA微调函数
    bnb_config=BitsAndBytesConfig(**BNB_CONFIG)#加载量化配置，**是字典解包，作用是QLoRA轻量化微调，节省内存
    #加载基座模型与分词器
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)#加载自动分词器，trust必须开启，因为Qwen模型需要本地执行自定义建模代码
    tokenizer.pad_token=tokenizer.eos_token#用结束符充当pad填充符，因为因果大模型默认没有padding符号
    tokenizer.padding_side="right"#填充文本在右边
    model=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)#加载模型，绑定4bit量化配置，计算精度使用bf16，自动分配硬件
    model=prepare_model_for_kbit_training(model)#适配量化模型训练，量化模型不能直接训练，该函数做底层适配：开启梯度，修正归一化层，兼容4bit权重的梯度反向传播
    #LoRA配置
    lora_config=LoraConfig(**LORA_CONFIG)#初始化LoRA配置，**是解包外部配置字典
    model=get_peft_model(model,lora_config)#将LoRA适配器插入，返回可微调LoRA模型
    model.print_trainable_parameters()#打印可训练参数量占总模型比例，LoRA一般仅0.1%-1%参数参与训练，用来验证LoRA是否生效
    #加载数据集
    train_dataset=load_dataset("json",data_files=TRAIN_DATA_PATH,split="train")#提前用脚本分开了train和val，所以这里写一样
    val_dataset=load_dataset("json",data_files=VAL_DATA_PATH,split="train")
    #训练参数
    train_args=TrainingArguments(**TRAIN_ARGS)#训练参数配置，解包外部字典**
    #SFT微调训练器
    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=train_args,
        dataset_text_field="text",#自动对text字段做分词处理
        max_seq_length=MAX_SEQ_LEN#单条文本最大长度阈值
    )
    #开始训练
    trainer.train()
    #保存最终LoRA权重
    trainer.save_model(LORA_SAVE_PATH)#权重保存路径
    print(f"LoRA训练完成，权重保存至:{LORA_SAVE_PATH}")
if __name__ == "__main__":#只有直接运行该脚本时，才会运行这个接口
    train_lora()#整套来量化，加载模型，训练流程

