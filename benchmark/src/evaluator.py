import json
from tqdm import tqdm
from .utils.text_cleaner import TextCleaner
from .utils.metrics import calculate_cer

class ASREvaluator:
    def __init__(self, output_path: str = "benchmark_results.jsonl"):
        self.output_path = output_path
        self.cleaner = TextCleaner()

    def evaluate(self, model_before, model_after, dataset, batch_size: int = 4):
        references = []
        before_predictions = []
        after_predictions = []

        print("\nStarting Benchmark...")
        # 清空先前的 JSONL 檔案
        with open(self.output_path, "w", encoding="utf-8") as f:
            pass

        num_samples = len(dataset)
        # 由於 tqdm 進度條在背景印出時需要保持清晰，我們將動態更新進度
        pbar = tqdm(range(0, num_samples, batch_size), desc="ASR Benchmark")

        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, num_samples)
            batch = dataset[start_idx:end_idx]

            audio_arrays = [sample["audio"] for sample in batch]
            sampling_rates = [sample["sampling_rate"] for sample in batch]
            batch_refs = [self.cleaner.clean(sample["reference"]) for sample in batch]

            # Gemma Before (Base) 推論
            before_outputs = model_before.transcribe_batch(audio_arrays, sampling_rates)
            before_preds = [self.cleaner.clean(pred) for pred in before_outputs]

            # Gemma After (Finetuned) 推論
            after_outputs = model_after.transcribe_batch(audio_arrays, sampling_rates)
            after_preds = [self.cleaner.clean(pred) for pred in after_outputs]

            # 收集與儲存結果
            for i in range(len(batch)):
                idx = start_idx + i
                ref_text = batch_refs[i]
                brz_pred = before_preds[i]
                gma_pred = after_preds[i]

                references.append(ref_text)
                before_predictions.append(brz_pred)
                after_predictions.append(gma_pred)

                # 即時印出辨識結果與對比
                print(f"\n[{idx+1}/{num_samples}]")
                print(f"  REF: {ref_text}")
                print(f"  GMA_BF: {brz_pred}")
                print(f"  GMA_AF: {gma_pred}")
                print("-" * 50)

                # 即時寫入 JSONL 存檔
                result_item = {
                    "index": idx,
                    "reference": ref_text,
                    "gemma_before": brz_pred,
                    "gemma_after": gma_pred
                }
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            # 計算累計的 CER 並顯示於進度條
            current_brz_cer = calculate_cer(references, before_predictions)
            current_gma_cer = calculate_cer(references, after_predictions)
            pbar.set_postfix({
                "BF_CER": f"{current_brz_cer*100:.2f}%",
                "AF_CER": f"{current_gma_cer*100:.2f}%"
            })

        # 計算最終 CER
        final_brz_cer = calculate_cer(references, before_predictions)
        final_gma_cer = calculate_cer(references, after_predictions)

        print("\n" + "="*20 + " FINAL RESULTS " + "="*20)
        print(f"Gemma-Before Overall CER : {final_brz_cer * 100:.2f}%")
        print(f"Gemma-After  Overall CER : {final_gma_cer * 100:.2f}%")
        print("="*55)
        
        return final_brz_cer, final_gma_cer
