import os
from dotenv import load_dotenv
from Agent.base_agent import generate_base_agent
from loguru import logger
import json
import asyncio
from Tools.data_profile_analysis import DataProfileAnalysis

load_dotenv('.env.dev')
API_KEY = os.getenv('API_KEY')
MODEL = os.getenv('MODEL')

with open('./Tools/generate_work_dir.py', 'r', encoding='utf-8') as f:
    code = f.read()
    
exec(code)

with open('./work_dir.txt', 'r', encoding='utf-8') as f:
    work_dir = f.read()
    
from Agent.code_executor_agent import coderagent
from Agent.modeler import run_modeler

async def workflow(user_input, data_path, work_dir):
    
    data_profile_analysis = DataProfileAnalysis(data_path, work_dir)
    data_report = data_profile_analysis.generate_report()
    
    prompt_template = """
    你是一个专业的数学建模助手，请基于以下上下文为用户提交的数学建模题目进行提供建模详细思路。注意一定要详细，并且你无需进行任何操作，只需要根据收到的信息给出建模策略。
    你需要根据用户给出的题目，使用arxiv-mcp-server工具参考4-6篇论文，并列出参考文献的网页链接
    如果无法从上下文中得到答案，请通过配置的fetch工具搜索一些专业的网页知识进行回答，并注意一定要标注参考网址的链接。若未参考网页内容则无需标注。
    过滤掉与问题极度相同的文献，防止查重率过高。
    回答格式为json。分别包含问题背景，问题分析和详细解题步骤，注意问题背景不是用户需要输入的内容，而是你需要从用户提供的问题中提炼的。
    必须严格按照示例输出格式输出，不能自己加内容，也不要输出任何解释性内容或询问性内容。
    
    数据文件路径:{data_path}
    
    数据报告:{data_report}

    用户问题: {question}

    上下文:
    {context}

    示例输出：
    {{
    "问题背景": "xxx，读取/未读取到数据",
    "问题一分析": "xxx，参考链接：",
    "问题二分析": "xxx，参考链接：",
    ……
    "详细解题步骤": {{
        "1、观察数据": "详细了解各个工作表的结构，内容，列名，索引等，可以按照需要了解数据正态性，缺失值，异常值等。注意一定要详细，特别是列名，索引，本步骤必须编写程序并调用代码执行工具求解完成",
        "2. 数据预处理：": "数据eda分析、可视化等，请在上一步观察结果的基础上进行可视化和eda分析，要保证可视化图片可以准确显示中文、负号等特殊字符",
        "3. 问题1求解:": "……",
        "4. 问题2求解:": "……",
        "5. 问题3求解:": "……",
        ……
        "6. 敏感性分析：": "……"
        }}
    }}
    """
    
    agent, prompt = generate_base_agent(prompt_template, MODEL, API_KEY, input_variables=["question", "context", "data_path", "data_report"])
    modeler = await run_modeler(agent, prompt, user_input, data_report, data_path)
    
    analysis_result = modeler['answer']
    
    with open(os.path.join(work_dir, 'analysis_result.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=4)
    logger.info(f'analysis_result: {analysis_result}')
    
    await coderagent.execute_task(
        problem_text=user_input,
        analysis_result=analysis_result,
        excel_path=data_path,
        work_dir=work_dir,
        data_report=data_report
    )
    
if __name__ == '__main__':
    user_input = '''
    C 题 农作物的种植策略
    根据乡村的实际情况，充分利用有限的耕地资源，因地制宜，发展有机种植产业，对乡村经济
    的可持续发展具有重要的现实意义。选择适宜的农作物，优化种植策略，有利于方便田间管理，提
    高生产效益，减少各种不确定因素可能造成的种植风险。
    某乡村地处华北山区，常年温度偏低，大多数耕地每年只能种植一季农作物。该乡村现有露天
    耕地 1201 亩，分散为 34 个大小不同的地块，包括平旱地、梯田、山坡地和水浇地 4 种类型。平旱
    地、梯田和山坡地适宜每年种植一季粮食类作物；水浇地适宜每年种植一季水稻或两季蔬菜。该乡
    村另有 16 个普通大棚和 4 个智慧大棚，每个大棚耕地面积为 0.6 亩。普通大棚适宜每年种植一季蔬
    菜和一季食用菌，智慧大棚适宜每年种植两季蔬菜。同一地块（含大棚）每季可以合种不同的作物。
    详见附件 1。
    根据农作物的生长规律，每种作物在同一地块（含大棚）都不能连续重茬种植，否则会减产；
    因含有豆类作物根菌的土壤有利于其他作物生长，从 2023 年开始要求每个地块（含大棚）的所有土
    地三年内至少种植一次豆类作物。同时，种植方案应考虑到方便耕种作业和田间管理，譬如：每种
    作物每季的种植地不能太分散，每种作物在单个地块（含大棚）种植的面积不宜太小，等等。2023
    年的农作物种植和相关统计数据见附件 2。
    请建立数学模型，研究下列问题：
    问题 1 假定各种农作物未来的预期销售量、种植成本、亩产量和销售价格相对于 2023 年保持
    稳定，每季种植的农作物在当季销售。如果某种作物每季的总产量超过相应的预期销售量，超过部
    分不能正常销售。请针对以下两种情况，分别给出该乡村 2024~2030 年农作物的最优种植方案，将
    结果分别填入 result1_1.xlsx 和 result1_2.xlsx 中（模板文件见附件 3）。
    (1) 超过部分滞销，造成浪费；
    (2) 超过部分按 2023 年销售价格的 50%降价出售。
    问题 2 根据经验，小麦和玉米未来的预期销售量有增长的趋势，平均年增长率介于5%~10%
    之间，其他农作物未来每年的预期销售量相对于 2023 年大约有±5%的变化。农作物的亩产量往往会
    受气候等因素的影响，每年会有±10%的变化。因受市场条件影响，农作物的种植成本平均每年增长
    5%左右。粮食类作物的销售价格基本稳定；蔬菜类作物的销售价格有增长的趋势，平均每年增长5%
    左右。食用菌的销售价格稳中有降，大约每年可下降1%~5%，特别是羊肚菌的销售价格每年下降幅
    度为5%。
    请综合考虑各种农作物的预期销售量、亩产量、种植成本和销售价格的不确定性以及潜在的种
    植风险，给出该乡村 2024~2030 年农作物的最优种植方案，将结果填入 result2.xlsx 中（模板文件见
    附件 3）。
    问题 3 在现实生活中，各种农作物之间可能存在一定的可替代性和互补性，预期销售量与销
    售价格、种植成本之间也存在一定的相关性。请在问题 2 的基础上综合考虑相关因素，给出该乡村
    2024~2030 年农作物的最优种植策略，通过模拟数据进行求解，并与问题 2 的结果作比较分析。
    附件 1 乡村现有耕地和农作物的基本情况
    附件 2 2023 年乡村农作物种植和相关统计数据
    附件 3 须提交结果的模板文件（result1_1.xlsx，result1_2.xlsx，result2.xlsx）
    '''
    data_path = r'D:\pythonproject\data_science\excel_files'
    
    asyncio.run(workflow(user_input, data_path, work_dir))
