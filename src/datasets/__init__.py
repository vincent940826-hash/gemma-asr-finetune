import os
import importlib

# 自動載入同目錄下的所有 dataset Python 模組
# 這將確保所有 Loader 的 @register_dataset 裝飾器都能在主程式執行時自動被調用並完成註冊
dataset_dir = os.path.dirname(__file__)
for file in os.listdir(dataset_dir):
    if file.endswith(".py") and file != "__init__.py" and file != "base.py":
        module_name = f"src.datasets.{file[:-3]}"
        importlib.import_module(module_name)
