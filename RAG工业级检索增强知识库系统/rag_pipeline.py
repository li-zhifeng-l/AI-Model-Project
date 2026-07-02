from vector_store import vector_retrieve#在文件里导出向量检索函数
from rerank import Reranker#在rerank文件导出重排工具类
from llm_generator import LLMGenerate#在llm_generator导入类包括模型加载，prompt拼接，文本生成推理类
from config import TOP_K,RERANK_TOP#全局配置文件导入TOP_K向量粗召回数量，RERANK_TOP重排后保留文档数量
#完整RAG流水线
def rag_chat(query:str):
    raw_chunks=vector_retrieve(query,TOP_K)#向量粗检索，调用之前导入的向量检索函数，传入query，TOP_K
    if not raw_chunks:#如果没有这个文本分块，返回未查询
        return "知识库未查询到相关信息"
    reranker=Reranker()#实例化重排类
    top_context=reranker.rerank(query,raw_chunks,RERANK_TOP)#Rerank精排，逐个计算用户问题真实语义相关性，过滤掉无关噪声片段，保留相关性最高的RERANK_TOP条作为参考上下文
    llm=LLMGenerate()#实例化语言大模型推理类
    prompt=llm.build_prompt(query,top_context)#调用类的函数，拼接RAG专用提示词
    answer=llm.generate(prompt)#调用类的函数，生成回答
    return {
        "query":query,
        "reference_context":top_context,
        "answer":answer#返回字典，问题，参考片段，答案
    }#llm组装prompt生成回答
#文档入库入口
def upload_knowledge(file_path):#定义知识库上传入库函数，是RAG系统知识库写入入口
    from utils import load_file#导入加载函数
    from vector_store import add_documents_to_db#导入入库函数
    docs=load_file(file_path)#加载出文档
    add_documents_to_db(docs)#将文档向量化持久化存入向量数据库

if __name__ == "__main__":#Python程序主入口判断，是本地直接运行脚本调试
    result=rag_chat("你的文档里的核心内容是什么？")#调用封装好的RAG全链路入口函数，传入测试问题
    print("回答:",result["answer"])#打印返回字典里的回答
    print("参考片段:",result["reference_context"])#打印字典里的参考片段






