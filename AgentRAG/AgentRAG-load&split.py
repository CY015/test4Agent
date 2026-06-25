# Document loading
## 基于SDK
from mineru import MinerU
import os

client = MinerU()
re = client.flash_extract('./test4loading.pdf')
re.save_markdown('./output/read1.md')

## 基于langchain
from langchain_mineru import MinerULoader

# 初始化客户端
loader = MinerULoader(
    source="./test4loading.pdf",
    mode="flash" # 可选: flash 、 precision
)
# 解析文档，返回值直接是LangChain的 Document集合
docs = loader.load()

# print(docs[0].metadata) # 元数据
# print(docs[0].page_content) # 文档内容

# 写到本地看看
# with open("./output/r5.md", "w", encoding="utf-8") as f:
#     f.write(docs[0].page_content)

# Text splitter
from langchain_text_splitters import CharacterTextSplitter
long_text = docs[0].page_content

# 固定切分
## 创建字符切分器
text_splitter=CharacterTextSplitter(
    separator="\n",     # 以换行符作为分隔
    chunk_size=1000,    # 每块最大1000字符
    chunk_overlap=200   # 块之间重叠200字符
)

## 切分文本
chunks = text_splitter.split_text(long_text)

print(f'原始文本长度: {len(long_text)} 字符')
print(f'切分为: {len(chunks)} 块')

for i, chunk in enumerate(chunks):
    print(f'--- Chunk {i+1} ({len(chunk)}字符) ---')
    print(chunk)
    print()


## 使用from_tiktoken_encoder，LangChain自带，无需额外安装tiktoken
token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",    # token分词器编码名
    chunk_size=1000,                # 每块最多1000 token
    chunk_overlap=200,              # 块之间重叠200字符
)

chunks = token_splitter.split_text(long_text)

print(f"原始文本长度: {len(long_text)} 字符")
print(f"切分为 {len(chunks)} 个块:\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)}字符) ---")
    print(chunk)
    print()



