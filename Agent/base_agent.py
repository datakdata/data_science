from langchain.prompts import PromptTemplate
from mcp_use import MCPAgent, MCPClient
from langchain_deepseek import ChatDeepSeek

def generate_base_agent(prompt_template, model, api_key, input_variables:list, temperature=0.3):
    
    llm = ChatDeepSeek(
    model=model,
    api_key=api_key,
    temperature=temperature,
    )
    
    client = MCPClient.from_config_file("./mcp.json")
    agent = MCPAgent(llm = llm, client = client)
    
    prompt = PromptTemplate(
    template=prompt_template,
    input_variables=input_variables
)
    return agent, prompt