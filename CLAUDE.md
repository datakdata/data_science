# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based data science modeling assistant system designed for mathematical modeling competitions. The system provides automated data analysis, modeling strategy generation, code execution, and academic paper reference retrieval.

## Core Architecture

- **Main Workflow** (`main.py`): Orchestrates the complete modeling pipeline
- **Agents** (`Agent/`): Specialized AI agents for different tasks
  - `base_agent.py`: Base agent class with MCP integration
  - `modeler.py`: Modeling strategy generator with RAG
  - `code_executor_agent.py`: Code execution with stateful workflow
- **Tools** (`Tools/`): Core functionality modules
  - `rag.py`: Retrieval-Augmented Generation for academic papers
  - `data_profile_analysis.py`: Automated data analysis and profiling
  - `code_interpreter.py`: Jupyter notebook code execution
  - `generate_work_dir.py`: Workspace management

## Key Dependencies

- Python ≥3.13
- LangChain ecosystem for agent orchestration
- FAISS for vector similarity search
- MCP (Model Context Protocol) for tool integration
- pandas/openpyxl for Excel data processing
- matplotlib/seaborn for visualization

## Development Commands

### Setup
```bash
uv sync  # Install dependencies
```

### Running the System
```python
python main.py  # Execute complete modeling workflow
```

### Configuration
- Environment variables in `.env.dev`:
  - `API_KEY`: DeepSeek API key
  - `MODEL`: Model name (e.g., "deepseek-chat")
  - `RETRIEVE_NUM`: Number of documents to retrieve (default: 10)
  - `EMBEDDING_MODEL`: Sentence transformer model for RAG
- MCP server configuration in `mcp.json`

## Workflow Process

1. **Data Analysis**: `DataProfileAnalysis` generates data profile reports
2. **Modeling Strategy**: `modeler.py` uses RAG to retrieve academic papers and generate modeling approaches
3. **Code Execution**: `code_executor_agent.py` executes step-by-step solutions with error handling
4. **Result Generation**: Outputs include JSON reports, Excel files, and solution code

## Important Directories

- `paper/`: Store PDF academic papers for RAG retrieval
- `excel_files/`: Input data files (Excel format)
- `workdir/`: Generated workspace with analysis results
- `faiss_index/`: Vector database for document retrieval

## Agent Integration

The system uses MCP (Model Context Protocol) to integrate external tools:
- `fetch`: Web search capabilities
- `arxiv-mcp-server`: Academic paper retrieval

## Code Execution

Code execution uses Jupyter notebooks via `NotebookCodeExecutor` with:
- Automatic error detection and retry mechanisms
- Stateful workflow management
- Step-by-step problem solving
- Result persistence in workspace directories