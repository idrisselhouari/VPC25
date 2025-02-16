import numpy as np
import librosa
import soundfile as sf

def anonymize(input_audio_path):
    """
    Anonymization algorithm
    
    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in one ".wav" format.
    
    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type np.float32.
    sr : int
        The sample rate of the processed audio.
    """
    
    # Load the source audio file
    audio, sr = load_audio(input_audio_path)
    
    # Apply Spectral Warping
    warped_audio = spectral_warping(audio, sr)
    
    # Apply Cepstral Transformation
    anonymized_audio = cepstral_transformation(warped_audio, sr)
    
    return anonymized_audio.astype(np.float32), sr

def load_audio(file_path, sr=16000):
    """ Load the audio file """
    audio, sample_rate = librosa.load(file_path, sr=sr)
    return audio, sample_rate

def spectral_warping(audio, sr=16000, warp_factor=1.2):
    """ Apply spectral warping to modify the frequency spectrum """
    D = librosa.stft(audio)
    warped_D = np.abs(D)**warp_factor * np.exp(1j * np.angle(D))
    warped_audio = librosa.istft(warped_D)
    return warped_audio

def cepstral_transformation(audio, sr=16000):
    """ Apply cepstral transformation (e.g., MFCC modification) """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_transformed = np.copy(mfcc)
    mfcc_transformed[0] += np.random.normal(0, 5, mfcc.shape[1])  # Add noise to first MFCC coefficient
    reconstructed_audio = librosa.feature.inverse.mfcc_to_audio(mfcc_transformed)
    return reconstructed_audio