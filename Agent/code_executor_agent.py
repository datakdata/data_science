from langgraph.graph import StateGraph
from Agent.base_agent import MCPAgent
from Tools.code_interpreter import NotebookCodeExecutor
from typing import Dict, Any, TypedDict
import json
from mcp_use import MCPClient
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
import sys
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import END, START
from langgraph.checkpoint.memory import MemorySaver

load_dotenv('.env.dev')
api_key = os.getenv('API_KEY')
model = os.getenv('MODEL')
max_retries = int(os.getenv('MAX_RETRY_COUNTS'))


class AgentState(TypedDict):
    problem: str
    analysis: dict
    excel_path: str
    data_overview: dict
    processed_steps: list
    sample_data: dict
    draft_code: str
    execution_result: dict   # 草稿代码执行结果
    error_info: str
    current_step: int        # 当前步骤序号
    total_steps: int         # 总步骤数
    step_results: dict={}       # 步骤执行结果 {step: result}
    text_result: str
    last_code: list
    work_dir: str
    data_report: str
    error_suggestion: str

class CodeExecutorAgent(MCPAgent):
        
    def create_execution_tool_metadata(self):
        """创建可序列化的工具元数据（符合OpenAI规范）"""
        return {
            "type": "function",  # 添加必需的type字段
            "function": {        # 嵌套工具定义在function字段下
                "name": "execute_python_code",
                "description": "执行Python代码并保存结果到Jupyter笔记本",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "要执行的Python代码"}
                    },
                    "required": ["code"]
                }
            }
        }
        
    def execute_and_save(self, code: str, notebook_name: str = None) -> dict:
        """执行代码并返回结果"""
        return self.executor.execute_and_save(code, notebook_name)
            
    def __init__(self):
        self.executor = NotebookCodeExecutor()
        # 可序列化的工具元数据列表
        self.tool_metadata = [self.create_execution_tool_metadata()]
        # 工具名到函数的映射
        self.tool_functions = {
            "execute_python_code": self.execute_and_save
        }
        self.workflow = self._build_workflow()
        self.llm = ChatDeepSeek(
            model=model,
            api_key=api_key,
            temperature=0.3,
            )
        
        self.process_count = 1

    def _build_workflow(self) -> StateGraph:
        """构建新的自主决策工作流"""
        workflow = StateGraph(AgentState)
        
        workflow.checkpointer = MemorySaver()
        
        # 定义节点
        workflow.add_node("preprocess_steps", self._preprocess_steps)
        workflow.add_node("step_processor", self._process_step)
        workflow.add_node("handle_error", self._handle_execution_error)
        workflow.add_node("finalize_results", self._finalize_results)
        
        # 初始流程
        # workflow.add_edge("inspect_data", "preprocess_steps")
        workflow.add_edge(START, "preprocess_steps")
        workflow.add_edge("preprocess_steps", "step_processor")
        
        # 核心决策流程
        workflow.add_conditional_edges(
            "step_processor",
            self._decide_next_step,
            {
                "next_step": "step_processor",
                "error_retry": "handle_error",
                "finalize": "finalize_results"
            }
        )
        
        # 错误处理循环
        workflow.add_edge("handle_error", "step_processor")
        workflow.add_edge("finalize_results", END)
        
        workflow.set_entry_point("preprocess_steps")
        return workflow.compile()
        
    def _decide_next_step(self, state: Dict) -> str:
        """决策下一步动作（增加边界检查）"""
        # 保护性检查：确保current_step和total_steps存在
        if "current_step" not in state or "total_steps" not in state:
            logger.error("状态中缺少current_step或total_steps键！")
            state.setdefault("current_step", 1)
            state.setdefault("total_steps", 1)
            return "error_retry"
            
        if state.get("error_info"):
            return "error_retry"
            
        # 边界检查：确保current_step不超过total_steps
        if state["current_step"] > state["total_steps"]:
            return "finalize"
            
        # 安全递增current_step
        # next_step = state["current_step"] + 1
        # if next_step > state["total_steps"]:
        #     logger.warning(f"步骤递增超出范围: {state['current_step']} -> {next_step} (总步数: {state['total_steps']})")
        #     return "finalize"

        return "next_step"
        
        
    def _handle_execution_error(self, state: Dict) -> Dict:
        """处理执行错误"""
        logger.warning(f"步骤 {state['current_step']} 执行出错，准备重试...")
        # 保留错误信息供LLM参考
        return state

    async def execute_task(self, problem_text: str, analysis_result: Dict,
                         excel_path: str, work_dir: str, data_report: str) -> Dict[str, Any]:
        state = {
            "problem": problem_text,
            "analysis": analysis_result,
            "excel_path": excel_path,
            "work_dir": work_dir,
            "data_report":data_report
        }
        config = {"configurable": {"thread_id": "1"}}
        
        try:
            # 首次执行并返回结果
            result = await self.workflow.ainvoke(state, config)
            if result and "final_execution_result" in result:
                return result["final_execution_result"]
            return result or {}
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            try:
                # 尝试从检查点恢复
                latest_state = await self.workflow.aget_state(config)
                if latest_state and latest_state.next != END:
                    logger.info(f"Recovering from checkpoint. Next node: {latest_state.next}")
                    result = await self.workflow.ainvoke(input=latest_state, config=config)
                    return result.get("final_execution_result", {})
                return {}
            except Exception as e:
                logger.error(f"Recovery failed: {str(e)}")
                return {"error": str(e)}


    def _preprocess_steps(self, state: Dict) -> Dict:
        """预处理步骤：合并子步骤到主步骤（增强状态安全性）"""
        # 创建新状态对象避免原始状态被意外修改
        raw_steps = dict(state["analysis"].get("详细解题步骤", {}))
        processed_steps = []
        
        # 处理步骤逻辑
        for title, process_method in raw_steps.items():
            single_step = {title: process_method}
            processed_steps.append(single_step)
        
        # 确保有默认步骤
        if not processed_steps:
            processed_steps = ["未找到解题步骤"]
            logger.warning("未解析出解题步骤，程序即将终止")
            sys.exit(1)
        
        # 更新状态
        state.update({
            "processed_steps": processed_steps,
            "current_step": 1,
            "total_steps": len(processed_steps)
        })
        
        # 添加调试日志
        logger.info(f"预处理完成: 生成{len(processed_steps)}个步骤 | 状态键: {list(state.keys())}")
        return state
    
    def _save_text_solution(self, content: str, work_dir: str, step: int):
        os.makedirs(work_dir, exist_ok=True)
        filename = f"step_{step}_solution.txt"
        path = os.path.join(work_dir, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"文本解决方案已保存至: {path}")
        return path


    def _process_step(self, state: Dict) -> Dict:
        """处理单个步骤：让LLM自主决策（添加状态保护）"""
        # 保护性检查：确保processed_steps存在
        # if "processed_steps" not in state:
        #     logger.error("状态中缺少processed_steps键！")
        #     state.setdefault("processed_steps", ["状态异常: 缺少解题步骤"])
        #     state.setdefault("current_step", 1)
        #     state.setdefault("total_steps", 1)
        
        # 添加状态调试日志
        logger.debug(f"步骤处理器状态键: {list(state.keys())}")
        logger.info(f"处理步骤 {state['current_step']}/{state['total_steps']}...")
        
        # 准备提示
        system_prompt = (
            "你是一个问题解决专家，请根据步骤描述决定解决方案。\n"
            "可选操作：\n"
            "1. 生成纯文本解决方案（无需代码）\n"
            "2. 生成并执行Python代码（使用execute_python_code工具），你自己不要在代码中加入任何调试信息\n"
            '''
            写代码时的其他注意事项：
            1、代码必须完整，不要出现“示例代码”、“假设已经读取了数据”，“假设……”， “由于代码较长，简化……”等情况，要严格根据题目信息和观察到的数据文件信息进行求解
            2、数据均为excel文件，请用pandas库进行读取，并注意读取到每个文件的所有工作表，包括隐藏工作表
            3、只专注于用户输入的当前步骤的描述的求解
            4、若遇到库缺失的问题，可以直接使用sys库执行库下载命令
            5、所有生成的结果均保存到当前文件夹
            6、写代码要参考用户传入的数据文件的分析报告
            7、进行数据可视化时，务必保证生成的图片能正常显示中文、负号等特殊符号
            8、不得简化模型和代码
            '''
            "注意：请勿解释决策过程，直接输出选择结果。"
        )
        
        # 获取当前步骤描述
        step_desc = state["processed_steps"][state["current_step"]-1]
        
        prompt_content = f"""
        ### 问题描述: 
        {state['problem']}
        
        ### 当前步骤 ({state['current_step']}/{state['total_steps']}): 
        当前需要解决的问题为：{step_desc}
        注意只解决当前问题
        
        请根据以前的运行结果修改报错的代码，不要一直用原来的代码。若是没有报错信息，则开始写新代码
        ### 上一问或上一次运行的代码（如有）:
        {state.get('last_code', '无')}
        
        ### 错误信息（如有）: 
        {state.get('error_info', '无')}
        
        ### 修正建议
        {state.get('error_suggestion', '无')}
        
        ### 上一问代码执行结果（如有）:注意不是让你抄上一问的代码，是让你知道上一问做了什么，便于后续操作
        {state.get('execution_result', '无')}
        
        ### 数据报告:
        {state['data_report']}
        
        ### 数据路径
        {state['excel_path']}
        """
        
        system_msg = SystemMessage(content=system_prompt)
        human_msg = HumanMessage(content=prompt_content)
        
        logger.info("调用LLM生成解决方案...")
        response = self.llm(
            messages=[system_msg, human_msg],
            tools=self.tool_metadata  # 使用可序列化的工具元数据
        )
        
        # 处理工具调用
        tool_calls = getattr(response, 'tool_calls', [])
        current_results = {}
        if tool_calls and tool_calls[0]['name'] == "execute_python_code":
            logger.info("检测到代码执行请求")
            # 执行代码路径
            code = tool_calls[0]['args']['code']
            draft_code = code
            last_code= code
            # 从工具映射中获取执行函数
            exec_func = self.tool_functions["execute_python_code"]
            notebook_name = f"{state['current_step']}-{self.process_count}.ipynb"
            exec_result = exec_func(code, notebook_name)
            
            if exec_result.get("status") == "success":
                self.process_count = 1
                error_info = ''
                # error_suggestion = ''
                # 执行成功时推进到下一步
                next_step = state["current_step"] + 1
                logger.success(f"步骤 {state['current_step']} 执行成功，推进到步骤 {state['current_step']+1}")
            else:
                self.process_count += 1
                next_step = state["current_step"]
                if self.process_count-1 > max_retries:
                    logger.error(f"步骤 {state['current_step']} 超过最大重试次数，程序终止")
                    sys.exit(1)
                else:
                    error_info = exec_result.get("output", "未知错误")
                    
                    # summarize_system_prompt = (
                    #     '''
                    #     你是一个python代码调试专家，请根据错误信息，给出修改建议，要求：
                    #     1、对于库确实的问题，给出的建议只能是建议使用其他库
                    #     2、明确指出代码错误的位置
                    #     3、给出的建议尽量简短，但是不能遗漏关键信息
                    #     '''
                    # )
                    
                    # summarize_human_msg = f'''{error_info}'''

                    # summarize_system_msg = SystemMessage(content=summarize_system_prompt)
                    # summarize_human_msg = HumanMessage(content=summarize_human_msg)

                    # logger.info("调用LLM生成代码修改建议...")
                    # summarize_response = self.llm(
                    #     messages=[summarize_system_msg, summarize_human_msg],
                    # )
                    
                    # error_suggestion = summarize_response.content
                
            # 保存代码结果
            current_results[state["current_step"]] = {
                "type": "code",
                "code": code,
                "result": exec_result
            }
            
        else:
            logger.info("检测到纯文本解决方案请求")
            error_info = ''
            draft_code = ''
            last_code = ''
            next_step = state["current_step"] + 1
            exec_result = current_results[state["current_step"]] = {
                "type": "text",
                "content": response.content
            }
            
            if work_dir := state.get("work_dir"):
                self._save_text_solution(str(exec_result), work_dir, state["current_step"])
            
        state.update({
            "step_results": current_results,
            "text_result": response.content,
            "error_info": error_info,
            "execution_result": exec_result,
            "draft_code": draft_code,
            "last_code": last_code,
            "current_step":next_step
        })
            
        return state
        
        
    async def _finalize_results(self, state: Dict) -> Dict:
        """汇总最终结果"""
        logger.info(f"所有{state['total_steps']}个步骤已完成")
        
        # 生成最终结果
        final_output = {
            "steps": state["step_results"],
            "requires_coding": any(result.get("type") == "code" for result in state["step_results"].values())
        }
        
        # 合并代码步骤
        if final_output["requires_coding"]:
            final_code = "\n\n".join(
                f"# 步骤 {step} 代码\n{result['code']}" 
                for step, result in state["step_results"].items() 
                if result.get("type") == "code"
            )
            final_output["final_code"] = final_code
            
        logger.success('任务执行完毕')
        
        return {**state, "final_execution_result": final_output}
    
coderagent = CodeExecutorAgent()
