
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Specify paths
audio_file = r"C:\\Users\\vijay\\Desktop\\whale\\audio_file.wav"
output_dir = r"C:\\Users\\vijay\\Desktop\\whale\\output_dir"

def process_audio_to_images(audio_file, output_dir, chunk_duration=3, sr=22050):
    """
    Splits an audio file into chunks and saves Mel spectrogram images for each chunk.

    Args:
    - audio_file (str): Path to the audio file.
    - output_dir (str): Path to save the spectrogram images.
    - chunk_duration (int): Duration of each chunk in seconds.
    - sr (int): Sampling rate for audio processing.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    print(f"Loading audio file: {audio_file}")
    y, original_sr = librosa.load(audio_file, sr=sr)
    print(f"Audio loaded. Duration: {len(y) / sr:.2f} seconds")

    # Calculate chunk length in samples
    chunk_length_samples = chunk_duration * sr
    total_chunks = len(y) // chunk_length_samples
    print(f"Processing {total_chunks} chunks...")

    for i in range(total_chunks):
        # Extract the chunk
        start_sample = i * chunk_length_samples
        end_sample = start_sample + chunk_length_samples
        chunk = y[start_sample:end_sample]
        
        # Create the Mel spectrogram
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Spectrogram - Chunk {i}")
        plt.tight_layout()
        
        # Save the plot as an image
        output_image_path = os.path.join(output_dir, f"chunk_{i}.png")
        plt.savefig(output_image_path)
        plt.close()  # Close the figure to avoid overlap
        print(f"Saved spectrogram: {output_image_path}")

    print(f"Processed {total_chunks} chunks and saved spectrogram images to {output_dir}")

# Call the function
process_audio_to_images(audio_file, output_dir)