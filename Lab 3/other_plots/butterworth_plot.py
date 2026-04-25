import numpy as np
import matplotlib.pyplot as plt

# Same signal as fft.py
fs = 1000
T = 0.5
t = np.arange(0, T, 1/fs)

freqs_sig = [10, 20, 30, 40, 50]
signal = np.zeros(len(t))
for i, f in enumerate(freqs_sig):
    signal += (i + 1) * np.cos(2 * np.pi * f * t)

lowcut = 5.0
highcut = 42.0

# Smooth bandpass using sigmoid transitions
w = np.linspace(0, 80, 10000)

def sigmoid(x, center, k):
    return 1 / (1 + np.exp(-k * (x - center)))

h_linear = sigmoid(w, lowcut, 0.8) * (1 - sigmoid(w, highcut, 0.8))
h_db = 20 * np.log10(np.clip(h_linear, 1e-10, None))

# FFT before and after (apply filter response at each frequency bin)
N = len(signal)
fft_freqs = np.fft.rfftfreq(N, 1/fs)
fft_orig = np.abs(np.fft.rfft(signal)) * 2 / N
fft_filt = fft_orig * sigmoid(fft_freqs, lowcut, 0.8) * (1 - sigmoid(fft_freqs, highcut, 0.8))

fig, (ax1, ax2) = plt.subplots(2, 1)

# --- Filter frequency response ---
ax1.plot(w, h_db, color='blue', label='Filter response')
ax1.axhline(-3, color='orange', linestyle='--', label='-3 dB')
ax1.axvline(lowcut, color='green', linestyle='--', alpha=0.45, label=f'Cutoff: {lowcut} / {highcut} Hz')
ax1.axvline(highcut, color='green', linestyle='--', alpha=0.45)
ax1.axvspan(lowcut, highcut, alpha=0.1, color='blue', label='Passband')
ax1.set_xlim(0, 80)
ax1.set_ylim(-70, 5)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Magnitude (dB)')
ax1.set_title(f'Bandpass Filter [{lowcut}, {highcut}] Hz')
ax1.legend()
ax1.grid()

# --- FFT before and after ---
offset = 0.4
ax2.stem(fft_freqs - offset, fft_orig, linefmt='blue', markerfmt='bo', basefmt=' ', label='Original')
ax2.stem(fft_freqs + offset, fft_filt, linefmt='red', markerfmt='ro', basefmt=' ', label='Filtered')
ax2.set_xlim(0, 70)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
ax2.set_title('FFT: Original vs Filtered (50 Hz peak attenuated)')
ax2.legend()
ax2.grid()

plt.tight_layout(pad=2.0)
plt.savefig('butterworth_plot.png')
plt.show()
