import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import os
import tempfile
import re
import ast
from datetime import datetime
from pathlib import Path

with open('./work_dir.txt', 'r', encoding='utf-8') as f:
    work_dir = f.read()

class NotebookCodeExecutor:
    def __init__(self, timeout=60, kernel_name='python3', output_dir=work_dir):
        """
        初始化代码执行器
        :param timeout: 单单元格执行超时时间(秒)
        :param kernel_name: Jupyter内核名称
        :param output_dir: 结果保存目录
        """
        self.timeout = timeout
        self.kernel_name = kernel_name
        self.output_dir = output_dir
        self.unsafe_commands = [
            'os.system', 'subprocess', 'shutil', 'sys.exit',
            'open', 'eval', 'exec', '__import__', 'rm ', 'del ',
            'while True:', 'fork', 'shutdown', 'kill'
        ]
        
        # 确保输出目录存在
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _is_code_safe(self, code: str) -> bool:
        """检查代码是否包含危险命令"""
        clean_code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)  # 移除注释
        for cmd in self.unsafe_commands:
            if cmd in clean_code:
                return False
        try:
            # 尝试解析语法树进行深层检查
            ast.parse(code)
        except SyntaxError:
            return False
        return True

    def execute_and_save(self, code: str, notebook_name=None) -> dict:
        """
        执行代码并保存结果到Jupyter笔记本
        :param code: 要执行的Python代码
        :param notebook_name: 笔记本文件名（忽略参数，使用固定文件名)
        :return: 包含执行结果和保存路径的字典
        """
        # 创建新的Notebook
        nb = nbformat.v4.new_notebook()
        nb.metadata.kernelspec = {
            "name": self.kernel_name,
            "display_name": f"Python 3 ({self.kernel_name})",
            "language": "python"
        }
        
        # 添加代码单元格
        code_cell = nbformat.v4.new_code_cell(source=code)
        nb.cells.append(code_cell)
        
        code_path = Path(self.output_dir) / 'executed_code'
        os.makedirs(code_path, exist_ok=True)
        save_path = Path(code_path) / notebook_name
        
        # 如果文件已存在，读取并追加新单元格
        if save_path.exists():
            existing_nb = nbformat.read(save_path, as_version=4)
            existing_nb.cells.append(code_cell)
            nb = existing_nb
        
        # 安全检查
        # if not self._is_code_safe(code):
        #     # 即使不执行也保存笔记本（不含输出）
        #     nbformat.write(nb, save_path)
        #     return {
        #         'status': 'security_error',
        #         'output': '代码包含潜在危险命令，拒绝执行',
        #         'notebook_path': str(save_path),
        #         'execution_count': 0
        #     }

        # 配置执行参数
        client = NotebookClient(
            nb,
            timeout=self.timeout,
            kernel_name=self.kernel_name,
            resources={'metadata': {'path': tempfile.gettempdir()}}
        )

        # 执行代码并处理结果
        try:
            last_cell_index = len(nb.cells) - 1
            
            client.execute()
            
            # 收集最后一个单元格的输出
            output = self._collect_outputs(nb.cells[last_cell_index])
            
            # 保存执行后的笔记本
            nbformat.write(nb, save_path)
            
            return {
                'status': 'success',
                'output': output,
                'notebook_path': str(save_path),
                'execution_count': nb.cells[0].execution_count
            }
        except CellExecutionError as e:
            # 保存包含错误信息的笔记本
            nbformat.write(nb, save_path)
            return {
                'status': 'execution_error',
                'output': str(e),
                'execution_count': nb.cells[0].get('execution_count', 0)
            }
        except Exception as e:
            return {
                'status': 'system_error',
                'output': f"执行失败: {str(e)}",
                'execution_count': 0
            }

    def _collect_outputs(self, cell) -> str:
        """收集所有输出并格式化为字符串"""
        outputs = []
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                outputs.append(output.text)
            elif output.output_type == 'execute_result':
                # 优先使用纯文本输出
                if 'text/plain' in output.data:
                    outputs.append(output.data['text/plain'])
                else:
                    outputs.append(str(output.data))
            elif output.output_type == 'error':
                outputs.append('\n'.join(output.traceback))
        return '\n'.join(outputs).strip()
    
    def get_execution_history(self, max_results=None) -> list:
        """获取执行记录(只返回固定文件的记录)"""
        history = []
        fixed_file = Path(self.output_dir) / "executed_code.ipynb"
        
        if fixed_file.exists():
            try:
                with open(fixed_file, 'r', encoding='utf-8') as f:
                    nb = nbformat.read(f, as_version=4)
                    if nb.cells:
                        cell_count = len(nb.cells)
                        history.append({
                            'notebook': fixed_file.name,
                            'path': str(fixed_file),
                            'timestamp': datetime.fromtimestamp(fixed_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'status': f"共{cell_count}个单元格",
                            'cell_count': cell_count
                        })
            except:
                pass
        return history

# 使用示例
if __name__ == "__main__":
    # 创建执行器，指定输出目录
    executor = NotebookCodeExecutor(output_dir=".")
    
    # 示例1: 成功执行并保存
    print("示例1: 成功执行数学计算")
    math_code = """
# 计算斐波那契数列
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# 打印前10个斐波那契数
print("斐波那契数列前10项:")
for i in range(10):
    print(fibonacci(i), end=' ')
    """
    result = executor.execute_and_save(math_code, "fibonacci_calculation")
    print(f"执行状态: {result['status']}")
    print(f"输出:\n{result['output']}")
    # print(f"保存路径: {result['notebook_path']}\n")