from langchain_ollama import OllamaEmbeddings

Oembed = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    dimensions=1024
)

# 向量化单条文本
text = "我爱上班"
vector = Oembed.embed_query(text)

print(f"文本: {text}")
print(f"向量维度: {len(vector)}")
print(f"向量前5维: {vector[:5]}")

# 批量向量化
texts = ["我要躺平", "我爱工作", "拒绝加班"]
vectors = Oembed.embed_documents(texts)
print(f"\n批量向量化: {len(vectors)} 条, 维度: {len(vectors[0])}")

import numpy as np

def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    return dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 比较向量相似度，值越小越相似
for v in vectors:
    similarity = cosine_similarity(vector, v)
    print("Cosine Similarity:", similarity)
