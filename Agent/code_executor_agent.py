from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from Tools.code_interpreter import NotebookCodeExecutor
from typing import Dict, Any, TypedDict, Annotated
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage
from loguru import logger
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool

load_dotenv('.env.dev')
api_key = os.getenv('API_KEY')
model = os.getenv('MODEL')
recursion_limit = os.getenv('RECURSION_LIMIT')

@tool
def execute_python_code(code: str, notebook_name: str = "solution.ipynb") -> str:
    """执行Python代码并保存结果到Jupyter笔记本
    
    Args:
        code: 要执行的Python代码
        notebook_name: 保存结果的笔记本文件名，默认为solution.ipynb，应当按照要求修改为step_n.ipynb（n是步骤数）
    """
    executor = NotebookCodeExecutor()
    result = executor.execute_and_save(code, notebook_name)

    if result.get("status") == "success":
        return f"代码执行成功: {result.get('output', '无输出')}"
    else:
        return f"代码执行失败: {result.get('output', '未知错误')}"
    
@tool
def save_text_solution(content: str, work_dir: str):
    """将无需写代码的题目的结果保存到工作目录
    
    Args:
        content: 需要保存的内容
        work_dir: 需要保存的位置，即工作目录
    """
    os.makedirs(work_dir, exist_ok=True)
    filename = f"solution.txt"
    path = os.path.join(work_dir, filename)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"文本解决方案已保存至: {path}")
    return path
    
tools = [execute_python_code, save_text_solution]

# 定义状态类型
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "对话消息历史"]
    problem: str
    analysis: Dict[str, Any]
    excel_path: str
    work_dir: str
    data_report: str


class CodeExecutorAgent:
    def __init__(self):
        self.llm = ChatDeepSeek(
            model=model,
            api_key=api_key,
            temperature=0.3,
        )

    def _get_react_prompt(self):
        """获取ReAct模式的系统提示"""
        return (
            "你是一个资深的数学建模代码手，请根据问题描述完整地解决整个问题。\n"
            "可选操作：\n"
            "1. 生成纯文本解决方案（无需代码）\n"
            "2. 生成并执行Python代码（使用execute_python_code工具），你自己不要在代码中加入任何调试信息\n"
            """
            写代码时的其他注意事项：
            1、代码必须完整，不要出现"示例代码"、"假设已经读取了数据"，"假设……"， "由于代码较长，简化……"等情况，要严格根据题目信息和观察到的数据文件信息进行求解
            2、数据均为excel文件，请用pandas库进行读取，并注意读取到每个文件的所有工作表，包括隐藏工作表
            3、完整地解决整个问题，不要只解决部分问题
            4、若遇到库缺失的问题，可以直接使用sys库执行库下载命令
            5、所有生成的结果均保存到当前文件夹
            6、写代码要参考用户传入的数据文件的分析报告
            7、进行数据可视化时，务必保证生成的图片能正常显示中文、负号等特殊符号
            8、不得简化模型和代码
            9、确保解决方案完整且准确
            10、问题分析中会分多个步骤，对于同一个步骤你要再同一个ipynb中写代码，命名格式为step_n.ipynb（n是步骤数）
            """
            "其中，题目必要信息如下："
        )
        

    async def execute_task(self, problem_text: str, analysis_result: Dict,
                         excel_path: str, work_dir: str, data_report: str) -> Dict[str, Any]:

        # 创建完整的任务提示
        task_prompt = f"""
            ### 问题描述:
            {problem_text}

            ### 题目分析:
            {analysis_result}

            ### 工作目录:
            {work_dir}

            ### 数据报告（注意数据报告中详细呈现了所有数据文件的数据结构，在进行求解时一定要参考这个报告）:
            {data_report}

            ### 数据路径
            {excel_path}

            ### 任务要求:
            请根据上述问题描述和数据报告，完整地解决整个问题。你可以：
            1. 生成纯文本解决方案（无需代码）
            2. 生成并执行Python代码（使用execute_python_code工具）

            请确保解决方案完整且准确。
            """
            
        
        react_agent = create_react_agent(
            self.llm,
            tools=tools,
            prompt=self._get_react_prompt()+task_prompt,
        )

        config = {"configurable": {"thread_id": f"task_{work_dir}", "recursion_limit": recursion_limit}}
        await react_agent.ainvoke(
            {"messages": [{"role": "user", "content": "请开始求解题目"}]},
            config=config
        )
                
        logger.success('任务执行完毕')
    
coderagent = CodeExecutorAgent()
