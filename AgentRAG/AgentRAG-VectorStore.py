from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 创建Embedding模型
ollama_embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",  # 性价比高的模型
    dimensions=1024  # 可选：减少维度以节省存储
)

# 创建向量库
vectorstore = Chroma(
    collection_name="example_collection",
    embedding_function=ollama_embeddings,
    persist_directory="./db/chroma_langchain_db",
)