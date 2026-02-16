import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from raspi_import import raspi_import
import scipy.signal as sig
import os

channels = 3
freqIn = 1000   #1 kHz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "day3-n172-5(FINAL).bin")

periodsCount = 4 #How many periods you want to display
periodTime = 1/freqIn 
rangePeriod = periodsCount * periodTime 

sample_period, data = raspi_import(DATA_PATH)

time_axis = np.arange(data.shape[0]) * sample_period

C = 3.3  # Vref = 3.3V
def converter(data): #Convert from counts to volts
    resulution = 2**12 - 1
    Vconv = (C/resulution * data)
    return Vconv

for i in range(channels):
    plt.plot(time_axis, converter(data[:, i]), label=f'ADC {i+1}')
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x*1e3:g}"))

plt.grid(True)
plt.xlim(0, rangePeriod)
plt.xlabel("Time [ms]")
plt.ylabel("Amplitude [V]")
plt.title(r"Sampled Sinusoidal of $f$=1 kHz")
plt.legend(loc='upper right')
plt.show()

#--------------------------PLOTTING OF FFT---------------------------------

x = converter(data[:, 0])
x_2 = converter(data[:, 1])
x_3 = converter(data[:, 2])
xh = x * np.hanning(len(x))
xb = x * np.blackman(len(x))
N = len(x)
Nfft = N*10   # Zero-padding factor

# Remove DC-offset

X = np.fft.fft(x, n=Nfft)
X_2 = np.fft.fft(x_2, n=Nfft)
X_3 = np.fft.fft(x_3, n=Nfft)
Xh = np.fft.fft(xh, n = Nfft)
Xb = np.fft.fft(xb, n = Nfft)
f = np.fft.fftfreq(Nfft, sample_period)

X = np.fft.fftshift(X)
X_2 = np.fft.fftshift(X_2)
X_3 = np.fft.fftshift(X_3)
Xh = np.fft.fftshift(Xh)
Xb = np.fft.fftshift(Xb)
f = np.fft.fftshift(f)

A = np.abs(X) / Nfft
A_2 = np.abs(X_2) / Nfft
A_3 = np.abs(X_3) / Nfft
A_dBFS = 20 * np.log10(A / np.max(A))
A_2dBFS = 20 * np.log10(A_2 / np.max(A_2))
A_3dBFS = 20 * np.log10(A_3 / np.max(A_3))
Ah = np.abs(Xh) / Nfft
Ah_dBFS = 20 * np.log10(Ah / np.max(Ah))
Ab = np.abs(Xb) / Nfft
Ab_dBFS = 20 * np.log10(Ab / np.max(Ab))

plt.figure()
plt.plot(f, A_dBFS, label=f'No Window')
plt.plot(f, A_2dBFS, label=f'ADC 2')
plt.plot(f, A_3dBFS, label=f'ADC 3')
#plt.plot(f, Ah_dBFS, label = f'Hanning Window')
#plt.plot(f, Ab_dBFS, label = f'Blackman Window')
plt.ylim(-130,10)
plt.legend()
plt.xlim(0,2000)
#plt.xlim(975,1025)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Relative Amplitude [dB]")
plt.title(r"Frequency Spectrum of the Sampled Sinusodal of $f$=1kHz")
plt.grid(True)
plt.show()


#-----------------------------PLOTTING OF CROSS CORRELATION---------------------


def crosscorr_delay(x, y, fs, max_lag=None):
    """
    Returns:
      n_delay: delay in samples (integer lag at max |cc|)
      lags: lag axis in samples
      cc: cross-correlation values
      lag_time: lag axis in seconds
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # remove DC
    x = x - np.mean(x)
    y = y - np.mean(y)

    cc = sig.correlate(x, y, mode="full")  # y vs x like prep code
    N = len(x)
    lags = np.arange(-N+1, N)

    if max_lag is not None:
        max_lag = int(max_lag)
        mask = (lags >= -max_lag) & (lags <= max_lag)
        cc = cc[mask]
        lags = lags[mask]

    idx = np.argmax(np.abs(cc))
    n_delay = int(lags[idx])

    lag_time = lags / fs 
    return n_delay, lags, cc, lag_time

x1 = converter(data[:, 0])
x2 = converter(data[:, 1])
x3 = converter(data[:, 2])

fs = 1/sample_period
max_lag = 10000

n21, lags21, cc21, t21 = crosscorr_delay(x2, x1, fs)
n31, lags31, cc31, t31 = crosscorr_delay(x3, x1, fs)
n32, lags32, cc32, t32 = crosscorr_delay(x3, x2, fs)

plt.plot(lags21, cc21, label=f"cc21 peak={n21} samp")
plt.plot(lags31, cc31, label=f"cc31 peak={n31} samp")
plt.plot(lags32, cc32, label=f"cc32 peak={n32} samp")
plt.xlabel("Lag (ms)")
plt.ylabel("Cross-correlation")
plt.grid(True)
plt.legend()
plt.show()

#------------------------DEVIATION OF CALCULATED ANGLE-------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "day3-n172-1(FINAL).bin")
sample_period, data1 = raspi_import(DATA_PATH)

DATA_PATH = os.path.join(BASE_DIR, "day3-n172-2(FINAL).bin")
sample_period, data2 = raspi_import(DATA_PATH)

DATA_PATH = os.path.join(BASE_DIR, "day3-n172-3(FINAL).bin")
sample_period, data3 = raspi_import(DATA_PATH)

DATA_PATH = os.path.join(BASE_DIR, "day3-n172-4(FINAL).bin")
sample_period, data4 = raspi_import(DATA_PATH)

DATA_PATH = os.path.join(BASE_DIR, "day3-n172-1(FINAL).bin")
sample_period, data5 = raspi_import(DATA_PATH)

fs = 1/sample_period
data = [data1, data2, data3, data4, data5]
all_real_degrees = []
for i in range(len(data)):
    x1 = data[i][:, 0]
    x2 = data[i][:, 1] 
    x3 = data[i][:, 2]

    n21, lags21, cc21, t21 = crosscorr_delay(x2, x1, fs)
    n31, lags31, cc31, t31 = crosscorr_delay(x3, x1, fs)
    n32, lags32, cc32, t32 = crosscorr_delay(x3, x2, fs)

    degreeRad = np.arctan2(np.sqrt(3)*(n31+n21), n31-n21+2*n32)
    degreeAngle = degreeRad*180/np.pi
    print(degreeAngle)
    all_real_degrees.append(degreeAngle)

all_real_degrees = np.array(all_real_degrees)
mean = np.mean(all_real_degrees)
s = np.sqrt(np.sum((all_real_degrees - mean)**2)/(len(all_real_degrees)-1))
s= np.std(all_real_degrees, ddof=1)
print("\nTheta Standardavvik: ", s)