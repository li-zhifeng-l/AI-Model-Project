import gradio as gr#Gradio是快速搭建AI Web可视化界面的Python库，无需前端JS/HTML，只用Python就能生成对话网页
import os#系统文件操作标准库
import subprocess#用于执行终端、命令行命令
from config import LORA_SAVE_PATH,MERGED_MODEL_PATH
from infer import qwen_infer
#推理接口
def chat_api(system,user,temp,top_p):#这是给Gradio网页提供上层交互接口函数
    if not os.path.exists(LORA_SAVE_PATH):#os.path.exists()判断磁盘上LoRA权重文件夹是否存在
        return "请先完成LoRA训练！"
    return qwen_infer(system,user,temp,top_p)
#训练启动接口
def start_train():#给Gradio网页提供一键启动训练按钮绑定函数
    subprocess.Popen(["python","train.py"])#subprocess.Popen:创建子进程执行系统命令，与run区别就是它可以异步后台运行，不阻塞当前Gradio网页服务主进程。传入列表等价于终端执行python train.py，启动LoRA训练脚本
    return "训练任务已后台启动，查看控制台日志！"
#合并模型接口
def run_merge():
    if not os.path.exists(LORA_SAVE_PATH):
        return "未检测到LoRA权重，无法合并"
    subprocess.run(["python","merge_lora.py"])#run同步阻塞执行，代码会卡住，必须等合并脚本全部跑完，但是合并操作很快，阻塞完全可接受
    return f"合并完成，完整模型路径:{MERGED_MODEL_PATH}"
#WebUI页面
with gr.Blocks(title="Qwen垂直领域LoRA微调系统") as demo:#gr.Blocks是Gradio最灵活的页面布局组件，比简单gr.Interface适合多功能平台。可以自由划分页面结构，也可以多标签页
    gr.Markdown("# Qwen LoRA轻量化微调平台")#页面顶部展示平台名称
    with gr.Tab("模型训练"):#标签页，第一个Tab标签页。还有gr.Row这是一行布局，里面所有横向摆放。gr.Column则是竖向一列布局
        train_btn=gr.Button("启动LoRA训练",variant="primary")#gr.Button是按钮组件
        train_info=gr.Textbox(label="训练状态")#gr.Textbox是文本展示框，用来接受函数返回的状态提示
        train_btn.click(start_train,outputs=[train_info])#.click绑定函数，outputs=[输出组件]
    with gr.Tab("模型合并"):#标签页，第二个Tab标签页
        merge_btn=gr.Button("合并LoRA基座")
        merge_info=gr.Textbox(label="合并结果")
        merge_btn.click(run_merge,outputs=[merge_info])
    with gr.Tab("对话测试"):#标签页,第三个Tab标签页
        sys_input=gr.Textbox(label="System提示词",value="你是垂直领域专业助手")
        user_input=gr.Textbox(label="用户问题")
        temp_slider=gr.Slider(minimum=0.01,maximum=1.0,value=0.7,label="temperature")#gr.Slider是滑动可视化调整生成参数，value是默认值
        top_p_slider=gr.Slider(minimum=0.1,maximum=1.0,value=0.9,label="top_p")
        submit_btn=gr.Button("生成回答")
        output_text=gr.Textbox(label="模型输出",lines=8)#lines是多行文本框
        submit_btn.click(chat_api,inputs=[sys_input,user_input,temp_slider,top_p_slider],outputs=[output_text])
if __name__ == "__main__":
    demo.queue()#开启请求队列，是大模型Gradio项目必备配置，大模型推理消耗久，防止并发请求同时抢占GPU导致程序崩溃
    demo.launch(server_name="0.0.0.0",server_port=8641,share=False)#demo.launch启动服务参数：server_name是网页网址，“0，0，0，0”允许局域网/其他服务器访问，不写只能本机127.0.0.1打开，server_port指定服务端口8641.share是关闭分享，内网部署场景关闭，保证安全
