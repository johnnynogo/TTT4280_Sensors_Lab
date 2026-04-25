import numpy as np
import matplotlib.pyplot as plt

# Signal parameters
fs = 1000
T = 0.5
t = np.arange(0, T, 1/fs)

# For sum of signal
freqs = [10, 20, 30, 40]
signal = np.zeros(len(t))
for i, f in enumerate(freqs):
    signal += (i + 1) * np.cos(2 * np.pi * f * t)

# Computing
N = len(signal)
fft_vals = np.abs(np.fft.rfft(signal)) * 2 / N
fft_freqs = np.fft.rfftfreq(N, 1/fs)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(t, signal, color='red')
ax1.set_xlim(0, T)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title(r'$\sum_{n=1}^{4} n\cos(n\omega t),\quad \omega = 10 \times 2\pi$')
ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax1.grid()

ax2.stem(fft_freqs, fft_vals, linefmt='blue', markerfmt='o', basefmt='k-')
ax2.set_xlim(0, 100)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')
ax2.set_title('FFT')
ax2.grid()

plt.tight_layout(pad=2.0)
plt.savefig('dft_plot_after.png')
plt.show()
