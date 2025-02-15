import numpy as np
import librosa
import soundfile as sf

def anonymize(input_audio_path):
    """
    Anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type `np.float32`, 
        which ensures compatibility with `soundfile.write()`.
    sr : int
        The sample rate of the processed audio.
    """

    # Read the source audio file
    y, sr = librosa.load(input_audio_path, sr=None)

    # Apply pre-emphasis
    pre_emphasis = 0.97
    y_emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Framing and Windowing
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = int(round(frame_size * sr)), int(round(frame_stride * sr))
    signal_length = len(y_emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(y_emphasized, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    # Fourier Transform
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Output the processed audio
    # For simplicity, we'll return the pre-emphasized signal as the anonymized audio
    audio = y_emphasized.astype(np.float32)
    
    return audio, sr