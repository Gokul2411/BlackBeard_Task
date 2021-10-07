import pyedflib
import numpy as np
import matplotlib.pyplot as plt

dt = 0.001                                                                      # 3000 points in 30 secs = 30/3000 = 0.001 secs

func_raw = pyedflib.EdfReader('SC4001E0-PSG.edf')                               # Reading .edf file
n = func_raw.signals_in_file
func_val = func_raw.readSignal(0)[:2500]                                        # Using 2500 data points for better visualization
numReadings = len(func_val)
func_fft = np.fft.fft(func_val)
lenFreqs = len(func_fft)
freq = (1 / (dt * lenFreqs)) * np.arange(lenFreqs)
PSD = (func_fft * np.conj(func_fft)) / numReadings                              # Getting the amplitude of the Power Spectrum

threshold = 20000

# Plotting Power Spectral Density

plt.plot(freq, PSD, 'r', label = 'PSD values')
plt.plot(np.arange(0, 1000, 1), [threshold for i in range(1000)], 'k--', label = "Threshold")
plt.xlim(0, 1000)
plt.title("Power Spectral Density vs frequency")
plt.legend()

# Removing all the frequencies with low threshold and applying inverse transform

indices = PSD > threshold
newPSD = PSD * indices
func_fft = func_fft * indices
func_filtered = (np.fft.ifft(func_fft))

# Final Plot

plt.figure()
plt.plot(func_val, 'c', label = "Raw Signal")
plt.plot(np.real(func_filtered), 'k', label = "Filtered Signal")
plt.title("Signal vs time")
plt.xlim(0, 2500)
plt.legend()
plt.show()