# 设置OpenMP线程数为8
import os
import time
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from typing import Any, List, Optional
from PIL import Image
from transformers import pipeline
import weaviate
import fitz  # PyMuPDF
from torchvision import datasets, transforms

# 从llama_index库导入相关类和函数
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery
import chromadb
from ipex_llm.llamaindex.llms import IpexLLM

class Config:
    """配置类,存储所有需要的参数"""
    model_path = "qwen2chat_int4"
    tokenizer_path = "qwen2chat_int4"
    question = "what is the FMD?"
    data_path = "./data/pdf"  # PDF文件目录
    image_data_path = "./data/images"  # 图像数据集路径
    persist_dir = "./chroma_db"
    embedding_model_path = "qwen2chat_src/AI-ModelScope/bge-small-zh-v1.5"
    max_new_tokens = 64

def load_vector_database(persist_dir: str) -> ChromaVectorStore:
    """
    加载或创建向量数据库
    
    Args:
        persist_dir (str): 持久化目录路径
    
    Returns:
        ChromaVectorStore: 向量存储对象
    """
    if os.path.exists(persist_dir):
        print(f"正在加载现有的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.get_collection("llama2_paper")
    else:
        print(f"创建新的向量数据库: {persist_dir}")
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = chroma_client.create_collection("llama2_paper")
    print(f"Vector store loaded with {chroma_collection.count()} documents")
    return ChromaVectorStore(chroma_collection=chroma_collection)

def load_pdf_data(data_dir: str) -> List[TextNode]:
    """
    加载并处理PDF数据
    
    Args:
        data_dir (str): PDF文件目录
    
    Returns:
        List[TextNode]: 处理后的文本节点列表
    """
    loader = PyMuPDFReader()
    text_parser = SentenceSplitter(chunk_size=384)
    nodes = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            documents = loader.load(file_path=file_path)

            for doc in documents:
                cur_text_chunks = text_parser.split_text(doc.text)
                for idx, text_chunk in enumerate(cur_text_chunks):
                    node = TextNode(text=text_chunk)
                    node.metadata = doc.metadata
                    nodes.append(node)
    
    return nodes

def load_image_data(image_data_path: str) -> List[TextNode]:
    """
    加载并处理图像数据集，将图像名称作为标签
    
    Args:
        image_data_path (str): 图像数据集路径
    
    Returns:
        List[TextNode]: 处理后的图像节点列表
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(image_data_path, transform=transform)
    image_nodes = []

    for img, label in dataset:
        img_name = dataset.imgs[label][0].split('/')[-1]  # 获取图像名称
        node = TextNode(text=img_name)  # 使用图像名称作为文本
        node.metadata = {"label": label, "image": img}
        image_nodes.append(node)
    
    return image_nodes

class VectorDBRetriever(BaseRetriever):
    """向量数据库检索器"""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        检索相关文档和图像
        
        Args:
            query_bundle (QueryBundle): 查询包
        
        Returns:
            List[NodeWithScore]: 检索到的文档和图像节点及其相关性得分
        """
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Retrieved {len(nodes_with_scores)} nodes with scores")
        return nodes_with_scores

def completion_to_prompt(completion: str) -> str:
    """
    将完成转换为提示格式
    
    Args:
        completion (str): 完成的文本
    
    Returns:
        str: 格式化后的提示
    """
    return f"\n</s>\n\n{completion}</s>\n\n"

def messages_to_prompt(messages: List[dict]) -> str:
    """
    将消息列表转换为提示格式
    
    Args:
        messages (List[dict]): 消息列表
    
    Returns:
        str: 格式化后的提示
    """
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"\n{message.content}</s>\n"
        elif message.role == "user":
            prompt += f"\n{message.content}</s>\n"
        elif message.role == "assistant":
            prompt += f"\n{message.content}</s>\n"

    if not prompt.startswith("\n"):
        prompt = "\n</s>\n" + prompt

    prompt = prompt + "\n"

    return prompt

def setup_llm(config: Config) -> IpexLLM:
    """
    设置语言模型
    
    Args:
        config (Config): 配置对象
    
    Returns:
        IpexLLM: 配置好的语言模型
    """
    return IpexLLM.from_model_id_low_bit(
        model_name=config.model_path,
        tokenizer_name=config.tokenizer_path,
        context_window=384,
        max_new_tokens=config.max_new_tokens,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        model_kwargs={},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="cpu",
    )

def predict_image(image_path: str) -> str:
    """
    预测图像类别
    
    Args:
        image_path (str): 图像路径
    
    Returns:
        str: 预测类别
    """
    # 加载预训练模型
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()
    
    # 预处理图像
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0)
    
    # 预测
    with torch.no_grad():
        preds = model(img)
    
    # 解析预测结果
    predicted_idx = torch.argmax(preds, dim=1).item()
    return predicted_idx

def main():
    """主函数"""
    config = Config()
    
    # 设置嵌入模型
    embed_model = HuggingFaceEmbedding(model_name=config.embedding_model_path)
    
    # 设置语言模型
    llm = setup_llm(config)
    
    # 加载向量数据库
    vector_store = load_vector_database(persist_dir=config.persist_dir)
    
    # 加载和处理PDF数据
    nodes = load_pdf_data(data_dir=config.data_path)
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
        vector_store.add([node])
    
    # 加载和处理图像数据
    image_nodes = load_image_data(config.image_data_path)
    for node in image_nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
        vector_store.add([node])
    
    # 设置查询
    query_str = config.question
    query_embedding = embed_model.get_query_embedding(query_str)
    
    # 执行向量存储检索
    print("开始执行向量存储检索")
    query_mode = "default"
    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )
    query_result = vector_store.query(vector_store_query)

    # 处理查询结果
    print("开始处理检索结果")
    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))
    
    # 设置检索器
    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=1
    )
    
    print(f"Query engine created with retriever: {type(retriever).__name__}")
    print(f"Query string length: {len(query_str)}")
    print(f"Query string: {query_str}")
    
    # 创建查询引擎
    print("准备与llm对话")
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    # 执行查询
    print("开始RAG最后生成")
    start_time = time.time()
    response = query_engine.query(query_str)

    # 打印结果
    print("------------RESPONSE GENERATION---------------------")
    print(str(response))
    print(f"inference time: {time.time()-start_time}")
