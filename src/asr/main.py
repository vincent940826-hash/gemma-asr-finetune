import asyncio
from src.asr.worker import BreezeStreamingASR

async def audio_producer(worker: BreezeStreamingASR):
    """
    Simulates incoming 20ms audio chunks from a WebSocket.
    """
    # Simulate 5 seconds of speech
    total_duration = 5.0
    chunk_duration = 0.02
    chunks = int(total_duration / chunk_duration)
    
    bytes_per_chunk = int(16000 * 2 * chunk_duration) # 640 bytes
    dummy_chunk = b'\x00' * bytes_per_chunk
    
    print("🎤 [Producer] Starting mock audio stream...")
    for i in range(chunks):
        # Simulate real-time arrival
        await asyncio.sleep(chunk_duration)
        
        is_end = (i == chunks - 1)
        await worker.push_audio(dummy_chunk, is_end=is_end)
    print("[Producer] Mock audio stream ended.")

async def event_consumer(worker: BreezeStreamingASR):
    """
    Consumes events from the worker and prints them.
    """
    while True:
        event = await worker.event_queue.get()
        if event['type'] == 'partial':
            print(f"🔄 [PARTIAL] {event['text']}")
        elif event['type'] == 'final':
            print(f"✅ [FINAL] {event['text']}")
            break

async def main():
    worker = BreezeStreamingASR()
    
    # Run producer and consumer concurrently
    producer_task = asyncio.create_task(audio_producer(worker))
    consumer_task = asyncio.create_task(event_consumer(worker))
    
    await asyncio.gather(producer_task, consumer_task)

if __name__ == "__main__":
    asyncio.run(main())
