import os
import pandas as pd

def append_csv(file_path, data, column_names):
    df = pd.DataFrame(data, index=column_names).T
    if not os.path.exists(file_path):
    # create a file if not exists
        df.to_csv(file_path, index=False)
    else:
        # append data to the file
        df.to_csv(file_path, mode='a', header=False, index=False)


def load_config(json_file_path):
    """加载JSON配置文件到命名空间对象"""
    from types import SimpleNamespace
    import json
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f, object_hook=lambda d: SimpleNamespace(**d))
