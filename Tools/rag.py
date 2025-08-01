import os
import json
import hashlib
import numpy as np
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
from loguru import logger
from dotenv import load_dotenv

load_dotenv('.env.dev')
model_name = os.getenv('EMBEDDING_MODEL')
work_dir = os.getenv('WORK_DIR')

class UnifiedRAGSystem:
    def __init__(self, paper_dir='./paper', index_path='./faiss_index'):
        """
        初始化 RAG 系统 - 专注于文档检索
        :param paper_dir: 文献存储目录
        :param index_path: 向量索引存储路径
        """
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )
        
        self.paper_dir = paper_dir
        self.global_index_path = index_path
        self.metadata_map = {}
        self.vectorstore = None
        
        # 检查并构建索引
        self._initialize_index()

    def _initialize_index(self):
        """初始化或加载索引"""
        # 记录文件夹中文献的数量
        papers_count = len(os.listdir(self.paper_dir) )
        
        # 检查文献数量是否有变化
        index_count_path = f'{self.global_index_path}_count.txt'
        if os.path.exists(index_count_path):
            with open(index_count_path, 'r') as f:
                old_papers_count = int(f.read())
        else:
            old_papers_count = 0
        
        if papers_count != old_papers_count:
            with open(index_count_path, 'w') as f:
                f.write(str(papers_count))
            logger.info(f"已记录当前文献数量: {papers_count}")
            logger.info("检测到文献数量变化，重新构建全局文献索引...")
            self._build_global_index()
            logger.info("全局文献索引构建完成")
        else:
            self._load_global_index()
            logger.info(f"已加载全局索引，包含 {self.vectorstore.index.ntotal} 个文档块")

    
    def _build_global_index(self):
        """构建统一的全局文献索引"""
        if not os.path.exists(self.paper_dir):
            os.makedirs(self.paper_dir, exist_ok=True)
            logger.info(f"创建了文献目录: {self.paper_dir}")
            return
        
        papers = [f for f in os.listdir(self.paper_dir) if f.endswith('.pdf')]
        all_chunks = []

        # 处理所有文献
        for paper in papers:
            try:
                loader = PyPDFLoader(os.path.join(self.paper_dir, paper))
                docs = loader.load()
                
                # 文本分割
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", "。", "！", "？", "。", "．"]
                )
                chunks = splitter.split_documents(docs)
                
                # 添加唯一标识符和元数据
                for chunk in chunks:
                    # 生成唯一ID (源文件名 + 页码 + 内容哈希)
                    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
                    page_num = chunk.metadata.get("page", 0) + 1
                    chunk_id = f"{os.path.splitext(paper)[0]}-{page_num}-{content_hash}"
                    
                    # 添加唯一ID到元数据
                    chunk.metadata["chunk_id"] = chunk_id
                    chunk.metadata["source"] = paper
                    chunk.metadata["page"] = page_num
                
                all_chunks.extend(chunks)
                logger.info(f"已处理 {paper}，添加 {len(chunks)} 个文本块")
            except Exception as e:
                logger.error(f"处理 {paper} 时出错: {str(e)}")
                continue
        
        if not all_chunks:
            logger.warning("没有找到可处理的PDF文献")
            return
        
        logger.info('开始保存全局索引')
        # 创建全局向量存储
        self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
        
        # 保存全局索引
        os.makedirs(os.path.dirname(self.global_index_path), exist_ok=True)
        self.vectorstore.save_local(self.global_index_path)
        
        # 保存元数据映射
        self.metadata_map = {
            doc.metadata["chunk_id"]: doc.metadata
            for doc in all_chunks
        }
        metadata_path = f'{self.global_index_path}_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_map, f, ensure_ascii=False, indent=2)
        
        logger.success(f"全局索引构建完成，包含 {len(all_chunks)} 个文本块")
    
    def _load_global_index(self):
        """加载全局索引和元数据"""
        # 加载向量存储
        self.vectorstore = FAISS.load_local(
            self.global_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # 加载元数据映射
        metadata_path = f'{self.global_index_path}_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata_map = json.load(f)
    
    def retrieve_documents(self, question: str, k: int = 5) -> List[Dict]:
        """
        检索相关文档 - RAG核心职责
        :param question: 用户问题
        :param k: 检索结果数量
        :return: 检索到的文档列表（包含内容和元数据）
        """
        if not self.vectorstore:
            raise RuntimeError("向量存储未初始化")
        
        # 执行相似度搜索
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # 格式化结果
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "未知来源"),
                "page": doc.metadata.get("page", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "")
            })
        
        return results

# 创建全局 RAG 工具实例
rag_tool = UnifiedRAGSystem()

def retrieve_research_documents(question: str, k: int = 5) -> List[Dict]:
    """
    研究文档检索工具 - RAG核心功能
    :param question: 用户问题
    :param k: 检索结果数量
    :return: 检索到的文档列表
    """
    return rag_tool.retrieve_documents(question, k)

async def generate_answer(agent, prompt, question: str, context_docs: List[Dict], data_path, data_report) -> Dict:
    """
    基于检索结果生成答案 - 上层调用者职责
    :param question: 用户问题
    :param context_docs: 检索到的文档列表
    :return: 包含答案和引用的字典
    """
    # 准备上下文
    context = ""
    references = []
    
    for i, doc in enumerate(context_docs):
        ref_id = f"ref-{i+1}"
        context += f"[{ref_id}] {doc['content']}\n\n"
        references.append({
            "id": ref_id,
            "chunk_id": doc['metadata'].get("chunk_id", ""),
            "paper": doc['source'],
            "page": doc['page'],
            "content_snippet": doc['content'][:100] + "...",
        })
    
    # 生成答案
    formatted_prompt = prompt.format(question=question, context=context, data_path=data_path, data_report=data_report)
    answer = await agent.run(formatted_prompt)
    
    try:
        # 尝试解析为JSON
        analysis_result = json.loads(answer)
    except json.JSONDecodeError:
        logger.error("无法解析答案为JSON格式")
        analysis_result = {"error": "答案格式不正确"}
    
    return {
        "answer": analysis_result,
        "references": references
    }

if __name__ == "__main__":
    rag_tool._initialize_index
