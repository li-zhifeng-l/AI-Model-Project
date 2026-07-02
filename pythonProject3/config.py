#全局配置
import torch.cuda

EMBED_MODEL_NAME="BAAI/bge-small-zh-v1.5"#向量化模型
RERANK_MODEL_NAME="BAAI/bge-reranker-base"#重排模型
LLM_MODEL_NAME="Qwen/Qwen1.5B-7B-Chat"#生成大模型
VECTOR_DB_PATH="./vector_store"
CHUNK_SIZE=384
CHUNK_OVERLAP=64
TOP_K=8
RERANK_TOP=3
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
ENCODE_KWARGS=True
COLLECITON_NAME="knowlage_base"
