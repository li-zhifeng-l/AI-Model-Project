import torch
#基座模型名称，自动拉取
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
#量化配置QLoRA 4bit
BNB_CONFIG={
    "load_in_4bit":True,#开启4bit量化加载，原生BF16:约13GB，4bit量化后约3GB
    "bnb_4bit_quant_type":"nf4",#量化数据类型，二选一：nf4和fp4，微调固定写nf4，贴合大模型权重分布
    "bnb_4bit_compute_dtype":torch.bfloat16,#计算用什么精度，推荐bfloat16，float16微调Loss容易爆炸
    "bnb_4bit_use_double_quant":True#双层压缩：进一步减少内存
}
#LoRA超参
LORA_CONFIG={
    "r":16,#r代表低秩矩阵的维度，决定LoRA拟合能力
    "lora_alpha":32,#缩放参数，alpha一般设为r的两倍
    "lora_dropout":0.05,
    "bias":"none",#训练偏置参数
    "target_modules":["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
#这些都是大模型（Llama / Mistral 等）里的核心线性层：
#q_proj, k_proj, v_proj：注意力的 QKV 投影层（最关键）
#o_proj：注意力输出投影
#gate_proj, up_proj, down_proj：FFN 前馈网络三层（让模型理解语义、逻辑）
    "task_type":"CAUSAL_LM"#任务类型，MASKED_LM掩码语言模型（预训练专用），SEQ_2_SEQ_LM编解码模型等
}
#训练参数
TRAIN_ARGS={
    "output_dir":"./weights/lora_outputs",#LoRA适配器保存路径
    "per_device_train_batch_size":2,#批次大小，显存不够降到2或1
    "gradient_accumulation_steps":4,#梯度累积，连续4步只计算梯度不更新参数，4步结束后统一更新一次权重
    "num_train_epochs":3,#完整遍历整个训练集3轮
    "learning_rate":2e-4,#LoRA标准学习率区间1e-4~3e-4，LoRA参数少，学习率要更大，全参数一般要1e-5，说白了就看步子迈多大
    "lr_scheduler_type":"cosine",#余弦学习率调度，相比线性下降收敛效果更好
    "warmup_ratio":0.1,#学习率预热，取全部训练总步数的前10%作为预热阶段，学习率从0线性慢慢提升，直到看见学习率设定值，预热阶段结束后，余弦退火，逐步降低学习率直到训练结束
    "optim":"adamw_8bit",#8bit优化器，梯度量化压缩存储，进一步降低显存占比
    "fp16":True,#混合精度训练，作用是加速训练
    "gradient_checkpointing":True,#梯度检查点，显存牺牲算力换显存，代价是训练速度轻微变慢，消费级显卡必开
    "logging_steps":10,#每训练10步，打印一次loss，学习率等
    "evaluation_strategy":"epoch",#评估策略，每跑完轮训练集使用val_dataset验证一次loss
    "save_strategy":"epoch",#保存策略同上
    "report_to":"none",#关闭第三方可视化工具
    "remove_unused_columns":True#自动删除Dataset里模型训练不需要的字段
}
#文本长度限制
MAX_SEQ_LEN=1024
#数据集路径
TRAIN_DATA_PATH="./data/processed_train.jsonl"
VAL_DATA_PATH="./data/processed_val.jsonl"
#权重路径
LORA_SAVE_PATH="./weights/lora_output/final_lora"
MERGED_MODEL_PATH="./weights/merged_model"#合并权重模型路径
