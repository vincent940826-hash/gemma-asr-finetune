import sys
# 強制繞過與 PyTorch 不相容的 torchcodec，促使 datasets 與 transformers fallback 至傳統解碼器
sys.modules['torchcodec'] = None

import argparse
import torch

from src.datasets.base import DATASET_REGISTRY
from src.models.breeze import BreezeASRModel
from src.models.gemma import GemmaASRModel
from src.evaluator import ASREvaluator

def main():
    parser = argparse.ArgumentParser(description="ASR Benchmark CLI for Breeze-ASR-25 and Gemma-4-E4B")
    
    # 支援註冊表動態選擇資料集
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        choices=list(DATASET_REGISTRY.keys()), 
        default="common_voice", 
        help="The dataset type to evaluate on"
    )
    # 若無手動指定，程式會自動使用對應 Loader 所定義的 DEFAULT_DATASET_ID
    parser.add_argument(
        "--dataset_id", 
        type=str, 
        default=None, 
        help="Optional HF dataset ID override"
    )
    # 若無手動指定，程式會自動使用對應 Loader 所定義的 DEFAULT_SPLIT
    parser.add_argument(
        "--split", 
        type=str, 
        default=None, 
        help="Optional dataset split override"
    )
    
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of samples to test (set to 0 or negative for all samples)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--output", type=str, default="benchmark_results.jsonl", help="Output JSONL file path")
    parser.add_argument("--model_breeze", type=str, default="MediaTek-Research/Breeze-ASR-25", help="Breeze model ID")
    parser.add_argument("--model_gemma", type=str, default="google/gemma-4-E4B-it", help="Gemma model ID")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 動態取得所選資料集的 Loader
    loader_cls = DATASET_REGISTRY[args.dataset_name]
    dataset_id = args.dataset_id or loader_cls.DEFAULT_DATASET_ID
    split = args.split or loader_cls.DEFAULT_SPLIT
    
    loader = loader_cls(dataset_id=dataset_id, split=split)
    
    # 載入資料
    max_samples = args.max_samples if args.max_samples > 0 else None
    dataset = loader.load(max_samples=max_samples)
    print(f"Loaded {len(dataset)} samples for evaluation from dataset: {args.dataset_name} ({dataset_id}, split: {split}).")

    # 2. 初始化模型
    breeze_model = BreezeASRModel(model_id=args.model_breeze, device=device)
    gemma_model = GemmaASRModel(model_id=args.model_gemma, device=device)

    # 3. 執行評估
    evaluator = ASREvaluator(output_path=args.output)
    evaluator.evaluate(
        breeze_model=breeze_model,
        gemma_model=gemma_model,
        dataset=dataset,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
