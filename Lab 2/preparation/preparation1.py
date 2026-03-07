import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Beskriv med Python-kode hvordan krysskorrelasjon kan brukes for å finne effektiv forsinkelse mellom
# to lydsignaler som er tatt opp med samplingsfrekvensen fs.
# Hint: du skal finne for hvilken forsinkelse som numpy.abs(krysskorrelasjonsfunksjonen) har maksimum.

N = 10000
fs = 4000
T = 1 / fs
t = np.arange(N) * T
t = t-np.mean(t)
# m = np.linspace(-fs, fs, N)
delay_samples = int(0.003 * fs)

x = np.sinc(100 * t)
y = np.roll(x, delay_samples)
cc = sp.signal.correlate(y, x, mode = 'full')

lags = np.arange(-N+1, N)
time_axis = lags * T
time_axis_ms = time_axis * 1000

idx = np.argmax(np.abs(cc))                     # which index returns largest value on func
print("Delay (samples):", lags[idx])
print("Delay (ms):", time_axis_ms[idx])

plt.figure()
plt.plot(t*1000, x, label='x(m)')
plt.plot(t*1000, y, label='y(m)')
plt.xlim(-75, 75)

plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(time_axis_ms, cc, label='zoomed in cc')
plt.xlim(-100, 100)

plt.figure()
plt.plot(time_axis_ms, cc, label='cc')
plt.xlim(-10, 10)

plt.grid(True)
plt.legend()
plt.show()

# np.abs(cc_function)