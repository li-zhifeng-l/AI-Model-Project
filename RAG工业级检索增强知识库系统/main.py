#FastAPI工程化接口
from fastapi import FastAPI,UploadFile#从fastapi框架导入FastAPI应用实例和文件上传类UploadFile
from rag_pipeline import rag_chat,upload_knowledge
import tempfile#导入临时文件tempfile，用来创建临时文件保存上传文档，处理完成后自动释放，避免占用磁盘
app=FastAPI(title="工业级RAG知识库系统")#创建Web服务主实例app
#问答接口
@app.get("/chat")#FastAPI的路由装饰器，。.get是一个GET请求接口，"/chat"是接口的访问路径：http://127.0.0.1:8000/chat，为什么用get不用post，get是简单字符串查询，若需传复杂上下文，则可改post
def chat(query:str):#这个接口的处理函数
    return rag_chat(query)
#文档上传入库接口
@app.post("/upload")#FastAPI的路由装饰器，注册POST接口，访问地址是/upload，必须用POST，因为GET无法携带文件二进制数据
async def upload(file:UploadFile):#async是异步函数，这样上传文件时性能更好。file:UploadFile接收前端上传的文件对象，FastAPI自动解析上传的文件
    with tempfile.NamedTemporaryFile(delete=False,suffix=file.filename) as tmp:#临时文件工具创建临时文件，不自动删除，文件后缀和原文件保持一致
        tmp.write(await file.read())#异步读取文件，把二进制写入本地临时文件
        tmp_path=tmp.name#获取临时文件在服务器的完整本地路径
    upload_knowledge(tmp_path)
    return {"msg":"文档入库完成"}
if __name__ == "__main__":
    import uvicorn#ASGI高性能异步Web服务器，专门用来运行FastAPI程序
    uvicorn.run(app,host="127.0.0.1",port=8000)#https//:127.0.0.1:8000/；host是本地回环地址，只能本机访问，port是服务监听接口

































































































