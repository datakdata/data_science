import os
from loguru import logger

def generate_workdir():
    """生成工作目录路径"""
    if not os.path.exists('./workdir'):
        os.mkdir('./workdir')
    dir_num = len(os.listdir('./workdir'))
    work_dir = f'./workdir/work_{dir_num}'
    os.mkdir(work_dir)
    
    with open('./work_dir.txt', 'w', encoding='utf-8') as f:
        f.write(work_dir)
    
    logger.info(f'工作目录创建完毕: {work_dir}')
    return work_dir

work_dir = generate_workdir()