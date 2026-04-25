import numpy as np
import matplotlib.pyplot as plt

f = np.linspace(0, 80, 10000)
lowcutoff = 5.0
highcutoff = 45.0
h = []
for fi in f:
    if lowcutoff <= fi <= highcutoff:
        h.append(0.0)
    else:
        h.append(-60.0)

plt.plot(f, h, color='blue', label='Ideal bandpass')
plt.axhline(-3, color='orange', linestyle='--', label='-3 dB')
plt.axvline(lowcutoff, color='green', linestyle='--', alpha=0.45, label=f'Cutoff: {lowcutoff} / {highcutoff} Hz')
plt.axvline(highcutoff, color='green', linestyle='--', alpha=0.45)
plt.axvspan(lowcutoff, highcutoff, alpha=0.1, color='blue', label='Passband')
plt.xlim(0, 80)
plt.ylim(-70, 5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title(f'Ideal Bandpass Filter [{lowcutoff}, {highcutoff}] Hz')
plt.legend()
plt.grid()
plt.savefig('bandpass_plot.png')
plt.show()
