import pandas as pd                                # For reading and handling CSV data
import numpy as np                                 # For numerical operations
from scipy.fft import fft, fftfreq                 # For performing FFT and computing frequencies

# Define the path to the CSV data file
file_path = "accel_data/accel_data_20250611_80.csv"

try:
    # Read the CSV file (no header assumed)
    df = pd.read_csv(file_path, header=None)

    # Assign column names to the dataframe
    df.columns = ['Timestamp(ms)', 'X', 'Y', 'Z']

    # Extract data from the dataframe
    timestamps = df['Timestamp(ms)'].values                 # Time in milliseconds
    x = df['X'].values.astype(float)                        # X-axis acceleration
    y = df['Y'].values.astype(float)                        # Y-axis acceleration
    z = df['Z'].values.astype(float)                        # Z-axis acceleration

    # Calculate the magnitude of the acceleration vector
    a_mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Compute the sampling rate (Hz) based on timestamp differences
    time_diff = np.mean(np.diff(timestamps))                # Average time interval in ms
    sample_rate = 1000 / time_diff                          # Convert to Hz (samples per second)

    # FFT analysis
    N = len(a_mag)                                          # Number of samples
    T = 1.0 / sample_rate                                   # Sampling period (s)
    yf = fft(a_mag - np.mean(a_mag))                        # Perform FFT (remove DC offset)
    xf = fftfreq(N, T)[:N // 2]                             # Frequency bins (one-sided)
    magnitude = 2.0 / N * np.abs(yf[:N // 2])               # Compute FFT amplitude spectrum

    # Identify the dominant frequency (peak in the spectrum)
    main_freq = xf[np.argmax(magnitude)]
    print(f"▶ Main vibration frequency: {main_freq:.2f} Hz")

except FileNotFoundError:
    print(f"Error: File not found → {file_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
