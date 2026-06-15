import time
import asyncio
import numpy as np
from faster_whisper import WhisperModel
from src.asr.agreement import get_longest_common_prefix

class BreezeStreamingASR:
    def __init__(self, model_path: str = "./models/breeze-25-ct2"):
        self.model_path = model_path
        # Initialize faster-whisper model
        self.model = WhisperModel(self.model_path, device="cuda", compute_type="float16")
        
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
            current_transcript = await self._generate(bytes(window), is_final=False)
            
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
            final_transcript = await self._generate(bytes(self.audio_buffer), is_final=True)
            
            await self.event_queue.put({"type": "final", "text": final_transcript})
            
            # Reset state for next utterance
            self.audio_buffer.clear()
            self.last_transcript = ""
            self.stable_prefix_emitted = ""
            self.last_inference_time = time.time()

    def _prepare_audio(self, audio_data: bytes) -> np.ndarray:
        """
        Converts raw PCM16 bytes into normalized numpy.float32 array.
        """
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    async def _generate(self, audio_data: bytes, is_final: bool = False) -> str:
        """
        Runs the actual faster-whisper inference in a background thread to avoid blocking.
        """
        audio_np = self._prepare_audio(audio_data)
        
        # Dynamic parameters based on inference type
        beam_size = 5 if is_final else 1
        condition_on_previous_text = False  # Set False for partial to avoid hallucinations
        
        loop = asyncio.get_running_loop()
        
        # Run the blocking transcribe call in a thread pool
        def transcribe_task():
            segments, info = self.model.transcribe(
                audio_np, 
                beam_size=beam_size, 
                condition_on_previous_text=condition_on_previous_text
            )
            return "".join(segment.text for segment in segments)
            
        transcript = await loop.run_in_executor(None, transcribe_task)
        return transcript.strip()
