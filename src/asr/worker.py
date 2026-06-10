import time
import asyncio
from src.asr.agreement import get_longest_common_prefix

class BreezeStreamingASR:
    def __init__(self, model_path: str = "MediaTek-Research/Breeze-ASR-25"):
        # Future-proofing: Structure assuming faster-whisper (CTranslate2) backend
        # e.g., self.model = faster_whisper.WhisperModel(model_path, device="cuda", compute_type="float16")
        self.model_path = model_path
        
        # Audio configuration
        self.sample_rate = 16000
        self.bytes_per_sample = 2 # PCM16
        
        # Buffers and state
        self.audio_buffer = bytearray()
        self.last_inference_time = time.time()
        self.last_transcript = ""
        self.stable_prefix_emitted = ""
        
        # Throttling
        self.inference_interval_sec = 0.7
        self.sliding_window_sec = 8.0
        self.sliding_window_bytes = int(self.sliding_window_sec * self.sample_rate * self.bytes_per_sample)
        
        # Queue for emitting events (partial, final)
        self.event_queue = asyncio.Queue()
        
        # To avoid concurrent inferences
        self._inference_lock = asyncio.Lock()

    async def push_audio(self, chunk: bytes, is_end: bool = False):
        """
        Receives raw PCM16 audio chunks.
        If is_end is True, simulates VAD end of utterance and triggers finalize.
        """
        self.audio_buffer.extend(chunk)
        
        if is_end:
            await self.finalize()
            return
        
        # Throttled periodic inference
        current_time = time.time()
        if current_time - self.last_inference_time >= self.inference_interval_sec:
            if not self._inference_lock.locked():
                # In a real faster-whisper scenario, inference blocks the CPU.
                # So we run it asynchronously to not block audio ingestion.
                asyncio.create_task(self._run_periodic_inference(current_time))

    async def _run_periodic_inference(self, trigger_time: float):
        async with self._inference_lock:
            self.last_inference_time = trigger_time
            
            # Extract sliding window
            window = self.audio_buffer[-self.sliding_window_bytes:]
            if len(window) == 0:
                return
            
            # Run inference
            current_transcript = await self._mock_generate(bytes(window))
            
            # Local Agreement
            stable_prefix = get_longest_common_prefix(self.last_transcript, current_transcript)
            
            # Only emit if prefix has grown
            if len(stable_prefix) > len(self.stable_prefix_emitted):
                self.stable_prefix_emitted = stable_prefix
                await self.event_queue.put({"type": "partial", "text": stable_prefix})
                
            self.last_transcript = current_transcript

    async def finalize(self):
        """
        Processes the entire current utterance buffer and resets the state.
        """
        async with self._inference_lock:
            if len(self.audio_buffer) == 0:
                return
            
            # Run inference on the full utterance
            final_transcript = await self._mock_generate(bytes(self.audio_buffer), is_final=True)
            
            await self.event_queue.put({"type": "final", "text": final_transcript})
            
            # Reset state for next utterance
            self.audio_buffer.clear()
            self.last_transcript = ""
            self.stable_prefix_emitted = ""
            self.last_inference_time = time.time()

    async def _mock_generate(self, audio_data: bytes, is_final: bool = False) -> str:
        """
        Mock inference mimicking faster-whisper.
        In reality, this would be:
        `await asyncio.get_running_loop().run_in_executor(None, self.model.transcribe, np_audio)`
        """
        # Simulate computation time
        await asyncio.sleep(0.1)
        
        # Dummy logic based on length of audio to simulate growing text
        duration_sec = len(audio_data) / (self.sample_rate * self.bytes_per_sample)
        
        # A mock transcript that grows over time
        base_text = "我目前正在參加一個計畫，需要你的協助，你是一名專業的工程師。"
        
        # Roughly 6 characters per second for our mock
        chars_to_show = int(duration_sec * 6)
        if chars_to_show == 0:
            return ""
            
        text = base_text[:chars_to_show]
        
        # Add some noise/flicker if it's not final, to test local agreement
        # This will randomly replace the last character, simulating an unstable tail.
        if not is_final and len(text) > 0 and chars_to_show < len(base_text):
            flicker = ["的", "了", "呢", "啊"][int(time.time() * 10) % 4]
            text = text[:-1] + flicker
            
        return text
