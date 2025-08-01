import pandas as pd
import os
from loguru import logger
from typing import Optional


class DataProfileAnalysis:
    
    def __init__(self, data_file:str, output_path:str):
        self.data_file = data_file
        self.output_path = output_path
        
    def list_data_names(self, data_file:str):
        """列出数据目录中的所有文件名"""
        if os.path.exists(data_file):
            data_dirs = os.listdir(data_file)
            logger.info(f"Data files in {data_file}: {data_dirs}")
            return data_dirs
        else:
            raise FileNotFoundError(f"Data file {data_file} does not exist.")
        
    def get_data_stats(self, df: pd.DataFrame) -> dict:
        """获取数据框的统计信息"""
        stats = {
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'head': df.head().to_dict('list')
        }
        
        # 检查每列的唯一值情况
        stats['unique_values'] = {}
        for col in df.columns:
            unique_vals = df[col].dropna().unique()
            # 转换为字符串并检查长度
            str_vals = [str(v) for v in unique_vals]
            if len(unique_vals) <= 10 and all(len(v) <= 10 for v in str_vals):
                stats['unique_values'][col] = unique_vals.tolist()
                
        return stats
        
    def list_data_columns(self, data_file:str) -> dict:
        """获取所有数据文件的列信息"""
        result = {}
        data_dirs = self.list_data_names(data_file)
        for data_dir in data_dirs:
            data_path = os.path.join(data_file, data_dir)
            data = pd.ExcelFile(data_path)
            sheet_names = data.sheet_names
            result[data_dir] = {}
            for sheet_name in sheet_names:
                df = data.parse(sheet_name)
                result[data_dir][sheet_name] = self.get_data_stats(df)
        return result
        
    def generate_report(self) -> str:
        """生成并保存数据分析报告"""
        report = "数据文件分析报告\n================\n\n"
        data_info = self.list_data_columns(self.data_file)
        
        for file_name, sheets in data_info.items():
            report += f"文件: {file_name}\n"
            for sheet_name, stats in sheets.items():
                report += f"工作表: {sheet_name}\n列信息:\n"
                for col, dtype in stats['dtypes'].items():
                    report += f"- 列名: {col}, 类型: {dtype}, 形状: {stats['shape']}\n"
                    if col in stats['unique_values']:
                        report += f"  类别: {stats['unique_values'][col]}\n"
                    report += f"  示例数据: {stats['head'][col][:5]}\n"
            report += "\n"
        
        if self.output_path:
            with open(os.path.join(self.output_path, 'data_profile_report.txt'), 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存至: {self.output_path}")
        
        return report


if __name__ == '__main__':
    data_path = r'D:\pythonproject\data_science\excel_files'
    
    data_profile = DataProfileAnalysis(data_path)
    report = data_profile.generate_report('./data_profile_report.txt')
