# Data Science Modeling Assistant System

## Project Overview
A Python-based assistant system for mathematical modeling competitions, providing:
- Automated data analysis and report generation
- AI-powered modeling strategy suggestions
- Code execution and result output
- Academic paper reference retrieval (via arxiv-mcp-server)

## System Architecture
```
data-science/
├── Agent/               # Agent modules
│   ├── base_agent.py    # Base agent class
│   ├── code_executor_agent.py  # Code execution agent
│   └── modeler.py       # Modeling strategy generator
├── Tools/               # Utility tools
│   ├── data_profile_analysis.py  # Data analysis
│   ├── generate_work_dir.py      # Workspace management
│   └── rag.py          # Retrieval-Augmented Generation
├── main.py             # Main workflow entry
├── pyproject.toml      # Project dependencies
└── mcp.json            # MCP server configurations
```

## Technology Stack
- Python ≥3.13
- Key Dependencies:
  - LangChain ecosystem
  - FAISS for vector search
  - pandas/openpyxl for data processing
  - matplotlib/seaborn for visualization
  - MCP server integration

## Installation
```bash
uv sync
```

## Configuration
1. Create `.env.dev` file with:
```ini
API_KEY=your_api_key_here
MODEL=your_model_name_here
```

2. Configure MCP servers in `mcp.json`

## Usage
Run the main workflow:
```python
python main.py
```

Before use, it is necessary to create a paper folder in the root directory to store references (in PDF format)

The system will:
1. Analyze input data
2. Generate modeling strategies
3. Execute solution code
4. Output results in JSON/Excel formats

## Example Workflow
Input: Mathematical modeling problem description  
Outputs:
- Data analysis report (`analysis_report.json`)
- Modeling strategy (`modeling_strategy.json`)
- Generated solution code (`solution.py`)
- Result files (`result*.xlsx`)

## License
[MIT License]
