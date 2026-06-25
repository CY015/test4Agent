from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 创建Embedding模型
ollama_embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",  # 性价比高的模型
    dimensions=1024  # 可选：减少维度以节省存储
)

# 创建向量库
# 将向量数据持久化到磁盘
vectorstore = Chroma(
    collection_name="example_collection",
    embedding_function=ollama_embeddings,
    persist_directory="./db/chroma_langchain_db",
)

# 创建递归切分器
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。"]  # 优先级从高到低
)

# 准备文档，我们用之前读取的Markdown文档来测试
with open("../output/r5.md", encoding="utf-8") as f:
    markdown_text = "\n".join(line for line in f.readlines())

# 用递归切分器切分文档
chunks = recursive_splitter.split_documents(
    [Document(page_content=markdown_text, metadata={"filename": "r5.md"})]
)
# 给文档生成id
# ids=[]
# for i,c in enumerate(chunks):
#     c.id = f"doc_{i+1}"
#     c.metadata['id'] = c.id
#     ids.append(c.id)
# # 删除旧文档
# vectorstore.delete(ids)
# # 添加新文档
# vectorstore.add_documents(chunks)

# 用户问题
query = "惠州园区的地址是什么"
print("相似度检索===================================================")
# 相似度检索
results = vectorstore.search(
    query=query,
    search_type="similarity",
    k = 2,
)

print(f"查询: {query}\n")
for i, doc in enumerate(results):
    print(f"结果 {i+1}: {doc.page_content}")
    # print(f"  元数据: {doc.metadata}")

print("基于metadata相似度检索===================================================")
# 基于metadata相似度检索
results = vectorstore.search(
    query=query,
    search_type="similarity",
    k = 2,
    filter={"id": "doc_1"}
)

print(f"查询: {query}\n")
for i, doc in enumerate(results):
    print(f"结果 {i+1}: {doc.page_content}")
    # print(f"  元数据: {doc.metadata}")

print("带相似度得分的检索===================================================")

# 带相似度得分的检索
results = vectorstore.similarity_search_with_relevance_scores(
    query=query,
    # search_type="similarity_score_threshold", 不需要search_type了
    score_threshold=0.1,
    k = 2
)

print(f"查询: {query}\n")
for doc, score in results:
    print(f"======文档: {doc.id}，得分：{score}=======")
    print(f"内容: {doc.page_content}")
    # print(f"元数据: {doc.metadata}")