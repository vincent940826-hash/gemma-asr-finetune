import abc

class BaseASRModel(abc.ABC):
    @abc.abstractmethod
    def transcribe_batch(self, audio_arrays: list, sampling_rates: list) -> list[str]:
        """
        Transcribe a batch of audio signals.
        Args:
            audio_arrays (list): List of numpy arrays containing the audio waveform data.
            sampling_rates (list): List of sampling rates corresponding to each audio signal.
        Returns:
            list[str]: The transcriptions for each audio signal.
        """
        pass
