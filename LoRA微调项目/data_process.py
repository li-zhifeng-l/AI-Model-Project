#数据处理脚本
import json#Python内置json；json.dumps()是将Python对象变成Json字符串；json.load()是将JSON字符串变成Python对象
import random#Python内置随机数库，打乱数据，常用于数据集划分，随机采样，打乱样本顺序
import jsonlines#.jsonl文件每一行单独是一个完整JSON对象
from tqdm import tqdm#循环进度条工具，观测代码运行进度。循环处理上万条样本时，自动生成可视化进度条
#Qwen官方对话模板封装
def format_qwen_prompt(system,user,assistant):#定义一个prompt格式化模板函数，system是系统提示词，user是用户提问内容，assistant模型历史回复内容
    text=f"<|im_start>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"#拼接指令模板<start>是对话开始符，另外是对话结束符，换行是用于分隔标识和文本内容
    return {"text":text}#把拼接完的对话存入字典
def process_raw_data(raw_path,train_out,val_out,val_radio=0.1):#定义处理原始对话数据集函数，raw_path是原始数据集文件路径即原始jsonl，train_out输出训练集文件路径，val_out输出验证集文件路径，val_radio是验证集比例，不传参则默认10%
    all_data=[]#用来存放格式化完成的样本，后续用来划分训练集和验证集
    #读取原始jsonl对话
    with jsonlines.open(raw_path,"r") as reader:#读取原始jsonl文件
        for item in tqdm(reader,desc="读取原始数据"):#带进度条循环遍历每条数据，desc是进度条前面标识
            conv=item["conversation"]#每条样本里conversation是一个列表
            sys_msg=""
            user_msg=""
            bot_msg=""#初始化三段文本空字符串
            for msg in conv:#遍历conversation列表，根据role，把对应内容存到三个变量
                if msg["role"]=="system":
                    sys_msg=msg["content"]
                elif msg["role"]=="user":
                    user_msg=msg["content"]
                elif msg["role"]=="assistant":
                    bot_msg=msg["content"]
            if sys_msg and user_msg and bot_msg:#布尔判断：如果三个全是非空，才保留这条信息
                formatted=format_qwen_prompt(sys_msg,user_msg,bot_msg)#调用模板处理信息
                all_data.append(formatted)#存入总数据集列表，等全部读取完再做数据集划分
    #划分训练验证集
    random.shuffle(all_data)#原地打乱数据列表，为什么要打乱？数据保持原始标注顺序，要是不打乱，则划分出内容类型差异很大
    val_size=int(len(all_data)*val_radio)#长度即个数乘于比例算出验证集的个数
    val_data=all_data[:val_size]#验证集样本取前val_size个
    train_data=all_data[val_size:]
    #保存文件
    with jsonlines.open(train_out,"w") as w:
        w.write_all(train_data)#一次性把列表内每条字典写成一行json
    with jsonlines.open(val_out,"w") as w:
        w.write_all(val_data)
    print(f"训练集：{len(train_data)}条，验证集:{len(val_data)}条")
if __name__ == "__main__":#区分脚本运行和模块导入
    import os#文件操作库，用来创建文件夹大，路径判断等
    os.makedirs("./data",exist_ok=True)#os.makedirs创建多层文件夹，./data目录，exist_ok防止文件夹已存在错误
    raw_file="./data/raw_data.jsonl"#原始未处理的对话数据集
    train_file="./data/processed_train.jsonl"#处理后输出的训练集
    val_file="./data/processed_val.jsonl"#处理后输出的验证集，这些路径跟config一致
    process_raw_data(raw_file,train_file,val_file)#调用处理原始数据集函数
