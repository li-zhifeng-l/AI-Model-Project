import re#Python内置的正则模块，用于文本清洗函数中使用，用来过滤水印，空格，换行，特殊乱码等，是文本降噪预处理
from langchain.text_splitter import RecursiveCharacterTextSplitter#LangChain文本分割器，是RAG项目主流分块器，负责递归字符分割
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader#文件加载器
from config import CHUNK_SIZE,CHUNK_OVERLAP#从config.py引入两个参数
#文本清洗：去除页眉页脚、乱码、多余多行。定位：文档识别后降噪文本的清洗方法
def clean_text(text:str)->str:#->声明函数完毕一定返回字符串
    text=re.sub(r"\n+","\n",text)#把多行连续换行替换成单个换行
    text=re.sub(r"\s+","",text)#把\s一串连续空白换成空字符串，意味着全部删掉；\s空格，\t制表符，\r回车
    text=re.sub(r"[水印|页眉|页脚|第\d+页]","",text)#去除水印等干扰文本
    return text.strip()#返回处理，清洗后输出的干净字符串
#语义滑动窗口分块(递归字符分割）
def get_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                          chunk_overlap=CHUNK_OVERLAP,
                                          separators=["\n\n","\n","。","，",""])#chunk_size：单个文本块允许最大字符长度，chunk_overlap：相邻两个文本块之间重叠的字符长度，防止关键语义被分割在两块中间，检索时丢失信息。separators是分割符列表，从前往后优先级，若是中文则优先使用中文标点符号。
#文档加载工具
def load_file(file_path:str):
    if file_path.endswith(".pdf"):
        loader=PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader=Docx2txtLoader(file_path)
    elif file_path.endswith(".txt"):
        loader=TextLoader(file_path)#通过后缀判断文件类型，匹配对应的LangChain加载器加载文档，外部只需要传文档路径file_path
    else:raise Exception("不支持该文件格式，仅支持txt，docs，pdf格式")#raise是抛出错误。是异常处理try/except的内容
    docs=loader.load()#调用加载器加载文档
    splitter=get_text_splitter()#调用之前封装的文本分割器
    split_docs=splitter.split_documents(docs)#执行分割，把文档按要求（预设块大小，重叠度，中文分割符）分成多个小块document列表
    for doc in split_docs:#循环遍历所有分块文档
        doc.page_content=clean_text(doc.page_content)#调用之前的清洗函数，去除脏数据
    return split_docs#若return在for循环内部，正常是该循环全部清洗完成后，在循环外统一返回


