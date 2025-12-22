from Tools.rag import retrieve_research_documents, generate_answer
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv('.env.dev')
RETRIEVE_NUM = int(os.getenv('RETRIEVE_NUM'))

async def run_modeler(agent, prompt, user_input: str, data_report, data_path: str = './data') -> str:
    """主应用逻辑"""
    logger.info("研究助理系统已启动。")
    
    # 步骤1: 检索相关文档
    logger.info("\n检索相关文档中...")
    retrieved_docs = retrieve_research_documents(user_input, k=RETRIEVE_NUM)
    logger.info(f"检索到 {len(retrieved_docs)} 个相关文档片段")
    
    # 步骤2: 基于检索结果生成答案
    logger.info("建模手建模中...")
    result = await generate_answer(agent, prompt, user_input, retrieved_docs, data_path, data_report)
    
    logger.success("\n建模手建模完成，开始求解题目")
    return result

if __name__ == "__main__":
    user_input = input("\n用户: ")
    run_modeler(user_input)
