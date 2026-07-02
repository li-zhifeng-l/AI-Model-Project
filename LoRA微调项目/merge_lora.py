#权重合并完整模型
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer#transformer是HuggingFace官方模型库
from peft import PeftModel#peft是高效微调库，来自HuggingFace.PeftModel是把LoRA增量权重挂载到原始基座模型上
from config import MODEL_NAME,LORA_SAVE_PATH,MERGED_MODEL_PATH#LORA_SAVE...是LoRA微调后权重保存目录，给PeftMoldel读取微调参数。MERGED_MODEL...是LoRA权重与基座重合并后的完整模型路径
def merge_lora_to_base():
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True)
    base_model=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )#加载出基座模型与分词器
    lora_model=PeftModel.from_pretrained(base_model,LORA_SAVE_PATH)#将LoRA权重挂载在基座模型
    #合并权重
    merged_model=lora_model.merge_and_unload()#merge_and_unload是PeftModel内置方法，merge融合，永久计算融合进基座Transformer的原始权重。融合完成后再unload卸载所有LoRA分支不再保留LoRA增量参数
    #保存完整模型
    merged_model.save_pretrained(MERGED_MODEL_PATH)#save_pretrained是transformer模型自带持久化保存方法
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    print(f"模型合并完成，完整模型保存至:{MERGED_MODEL_PATH}")
if __name__ == "__main__":#只有直接运行这个.py才会执行这个函数
    merge_lora_to_base()
