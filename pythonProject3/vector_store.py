import chromadb#原生向量库，LangChain的Choroma只是上层封装，真正向量存储，相似度检索，持久化，索引管理都是它提供的
from langchain.vectorstores import Chroma#上层封装，自动对接HuggingFaceEmbedding，提供入库，相似度检索，转检索器接入问答链
from langchain.embeddings import HuggingFaceEmbeddings#加载HuggingFace上开源文本向量化模型，统一管理里向量化
from config import EMBED_MODEL_NAME,VECTOR_DB_PATH,DEVICE,ENCODE_KWARGS,COLLECITON_NAME
#初始化Embedding
def get_embedding_model():
    embeddings=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME,
                                     model_kwargs={"device":DEVICE},
                                     encode_kwargs={"normalize_embeddings":ENCODE_KWARGS})#encode_kwargs是文本编码生成向量的参数，然后开启归一化
    return embeddings
#初始化向量库
def get_vector_store():
    embedding=get_embedding_model()
    client=chromadb.PersistentClient(path=VECTOR_DB_PATH)#chromadb.Persi..是持久化客户端
    db=Chroma(client=client,
              collection_name=COLLECITON_NAME,
              embedding_function=embedding)#client绑定持久化底层客户端，collection_name是向量集合名称，相当于数据库的一张表。embedding_function绑定向量模型
    return db
#新增文档入库
def add_documents_to_db(docs):
    if not docs:
        print("无有效文本块，跳过入库")
        return
    try:#捕获入库阶段可能出现的异常，全局捕获，只要try内部任意步骤出错，则进入except分支，打印错误信息
        db = get_vector_store()
        db.add_documents(docs)#是LangChain内置入库方法，自动执行文本向量化，写入本地Choroma
        print(f"成功入库{len(docs)}个文本块")
    except Exception as e:
        print(f"入库失败，错误信息：{str(e)}")
#粗检索：向量相似度召回
def vector_retrieve(query:str,top_k:int):#传入用户提问文本query和需要召回的片段数量top_k，是RAG检索函数
    db=get_vector_store()
    res=db.similarity_search_with_score(query,k=top_k)#同时返回文档对象和相似度分数
    chunks=[]
    for doc,score in res:
        chunks.append({"text":doc.page_content,#知识库原文文本
                       "source":doc.metadata,#文档元数据
                       "sim_score":score})#相似度得分
    return chunks#文本分块

