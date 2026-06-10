import json
from tqdm import tqdm
from .utils.text_cleaner import TextCleaner
from .utils.metrics import calculate_cer

class ASREvaluator:
    def __init__(self, output_path: str = "benchmark_results.jsonl"):
        self.output_path = output_path
        self.cleaner = TextCleaner()

    def evaluate(self, breeze_model, gemma_model, dataset, batch_size: int = 4):
        references = []
        breeze_predictions = []
        gemma_predictions = []

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

            # Breeze ASR 推論
            breeze_outputs = breeze_model.transcribe_batch(audio_arrays, sampling_rates)
            breeze_preds = [self.cleaner.clean(pred) for pred in breeze_outputs]

            # Gemma ASR 推論
            gemma_outputs = gemma_model.transcribe_batch(audio_arrays, sampling_rates)
            gemma_preds = [self.cleaner.clean(pred) for pred in gemma_outputs]

            # 收集與儲存結果
            for i in range(len(batch)):
                idx = start_idx + i
                ref_text = batch_refs[i]
                brz_pred = breeze_preds[i]
                gma_pred = gemma_preds[i]

                references.append(ref_text)
                breeze_predictions.append(brz_pred)
                gemma_predictions.append(gma_pred)

                # 即時印出辨識結果與對比
                print(f"\n[{idx+1}/{num_samples}]")
                print(f"  REF: {ref_text}")
                print(f"  BRZ: {brz_pred}")
                print(f"  GMA: {gma_pred}")
                print("-" * 50)

                # 即時寫入 JSONL 存檔
                result_item = {
                    "index": idx,
                    "reference": ref_text,
                    "breeze_prediction": brz_pred,
                    "gemma_prediction": gma_pred
                }
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            # 計算累計的 CER 並顯示於進度條
            current_brz_cer = calculate_cer(references, breeze_predictions)
            current_gma_cer = calculate_cer(references, gemma_predictions)
            pbar.set_postfix({
                "BRZ_CER": f"{current_brz_cer*100:.2f}%",
                "GMA_CER": f"{current_gma_cer*100:.2f}%"
            })

        # 計算最終 CER
        final_brz_cer = calculate_cer(references, breeze_predictions)
        final_gma_cer = calculate_cer(references, gemma_predictions)

        print("\n" + "="*20 + " FINAL RESULTS " + "="*20)
        print(f"Breeze-ASR-25 Overall CER : {final_brz_cer * 100:.2f}%")
        print(f"Gemma-4-E4B    Overall CER : {final_gma_cer * 100:.2f}%")
        print("="*55)
        
        return final_brz_cer, final_gma_cer
