import abc

# 全域資料集註冊表
DATASET_REGISTRY = {}

def register_dataset(name):
    """將資料集類別註冊到註冊表中的裝飾器"""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

class BaseDatasetLoader(abc.ABC):
    DEFAULT_DATASET_ID = None
    DEFAULT_SPLIT = None

    def __init__(self, dataset_id: str = None, split: str = None):
        self.dataset_id = dataset_id or self.DEFAULT_DATASET_ID
        self.split = split or self.DEFAULT_SPLIT

    @abc.abstractmethod
    def load(self, max_samples: int = None) -> list[dict]:
        """
        載入並格式化資料集，回傳統一的樣式：
        [
          {"audio": np.ndarray, "sampling_rate": int, "reference": str},
          ...
        ]
        """
        pass
