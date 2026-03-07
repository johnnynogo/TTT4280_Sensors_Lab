import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

"""
Transmitans:
??
69
70
61
62

Reflektans
62
58
56
56
54

Blame it on:
Cold hands, he's 'brown', high pulse, breathing

Yes, coldness makes it significantly more difficult to measure pulse, 
particularly when using optical sensors (like smartwatches) or pulse oximeters
Cold temperatures cause vasoconstriction, a process where blood vessels near the 
skin narrow to conserve body heat. This reduces blood flow and circulation to 
the extremities (fingers and wrists), making it hard for devices to detect a 
strong pulse signal. (interpreted as lower than it is basically)

Maybe check if red is better than green in some cases for measuring 
pulse? could explain wrong measurements
"""



# Load the file
df = pd.read_csv("reflektans_1_roi (THROWAWAY).csv", sep=r"\s+", header=None)
df.columns = ["RED", "GREEN", "BLUE"]

# Time axis
time = np.linspace(0, 30, len(df))
dt = time[1] - time[0]          # sampling interval
N = len(df)                     # number of samples

#==============TIME DOMAIN PLOT=================#
for col in df.columns:
    plt.plot(time, df[col], label=col)

plt.title("Pulse for each RGB")
plt.xlabel("Seconds")
plt.ylabel("Energy?")
plt.legend()
plt.grid(True)
plt.show()

#=================FFT====================#
fft_mag_all = []
freqs_all = []
for col in df.columns:
    signal = df[col].to_numpy()

    # Remove mean
    signal_demeaned = signal - np.mean(signal)

    fft_vals = np.fft.fft(signal_demeaned)
    freqs = np.fft.fftfreq(N, d=dt)

    # Keep only positive frequencies
    mask = (freqs >= 0) & (freqs <= 6)
    freqs_pos = freqs[mask]
    fft_mag = np.abs(fft_vals[mask]) #For magnitude !!!!!
    fft_mag_all.append(fft_mag)
    freqs_all = freqs_pos

    plt.plot(freqs_pos, fft_mag, label = f"FFT of {col} (mean removed)")

plt.legend()    
plt.title("FFT")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

#================FFT WITH FILTER=================#
Nfft = N*8
fs = 40
lower_limit = 0.5
upper_limit = 4.0
b, a = butter(3, [lower_limit/(fs/2) , upper_limit/(fs/2)], btype='band')
window = np.hanning(N)

fft_mag_filtered_all = []
freqs_filtered_all = []
for col in df.columns:
    signal = df[col].to_numpy()

    # Remove mean
    signal_demeaned = signal - np.mean(signal)
    signal_filtered = filtfilt(b, a, signal_demeaned * window)

    fft_vals = np.fft.fft(signal_filtered, n=Nfft)
    freqs = np.fft.fftfreq(Nfft, d=dt)

    # Keep only positive frequencies
    mask = (freqs >= 0) & (freqs <= 6)
    freqs_pos = freqs[mask]
    fft_mag = np.abs(fft_vals[mask]) #For magnitude !!!!!
    fft_mag_filtered_all.append(fft_mag)
    freqs_filtered_all = freqs_pos

    plt.plot(freqs_pos, fft_mag, label = f"FFT of {col} (mean removed)")

plt.legend()    
plt.title("FFT Filter Removed")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()


#===================FIND PULSE==================#
def find_pulsefrequency(magnitude, freq):
    relevant_indices = np.where((freq >= lower_limit) & (freq <= upper_limit))
    relevant_magnitude = magnitude[relevant_indices]
    relevant_freq = freq[relevant_indices]

    if len(relevant_magnitude) == 0:
        return None  # Ingen relevante frekvenser funnet

    max_index = np.argmax(relevant_magnitude)
    pulsefrequency = relevant_freq[max_index]
    
    return pulsefrequency

#--Before filter--
pulsefrequency_red_before = find_pulsefrequency(fft_mag_all[0], freqs_all)
pulsefrequency_green_before = find_pulsefrequency(fft_mag_all[1], freqs_all)
pulsefrequency_blue_before = find_pulsefrequency(fft_mag_all[2], freqs_all)
print(f"Pulsfrekvens før filter (Rød): {pulsefrequency_red_before * 60:.2f} BPM")
print(f"Pulsfrekvens før filter (Grønn): {pulsefrequency_green_before * 60:.2f} BPM")
print(f"Pulsfrekvens før filter (Blå): {pulsefrequency_blue_before * 60:.2f} BPM")

#--After filter--
pulsefrequency_red = find_pulsefrequency(fft_mag_filtered_all[0], freqs_filtered_all)
pulsefrequency_green = find_pulsefrequency(fft_mag_filtered_all[1], freqs_filtered_all)
pulsefrequency_blue = find_pulsefrequency(fft_mag_filtered_all[2], freqs_filtered_all)
print(f"\nPulsfrekvens etter filter (Rød): {pulsefrequency_red * 60:.2f} BPM")
print(f"Pulsfrekvens etter filter (Grønn): {pulsefrequency_green * 60:.2f} BPM")
print(f"Pulsfrekvens etter filter (Blå): {pulsefrequency_blue * 60:.2f} BPM")

#===================SNR==================#
pulse_real = 62  
Pulse_freq = pulse_real / 60  #Example: 60 BPM is 1 Hz
Measured_pluse_freq = pulsefrequency_green  

freq_tolerance = 0.1  # In Hz

#[pulse_frequency - frequency_tolerance, .., .., pulse_frequency + frequency_tolerance]
signal_indices_before = np.where((freqs_all >= Measured_pluse_freq - freq_tolerance) & 
                          (freqs_all <= Measured_pluse_freq + freq_tolerance))[0]

signal_indices_filtered = np.where((freqs_filtered_all >= Measured_pluse_freq - freq_tolerance) & 
                          (freqs_filtered_all <= Measured_pluse_freq + freq_tolerance))[0]

power_red_before = (fft_mag_all[0]**2) / N**2
power_green_before = (fft_mag_all[1]**2) / N**2
power_blue_before = (fft_mag_all[2]**2) / N**2

power_red_filtered = (fft_mag_filtered_all[0]**2) / N**2
power_green_filtered = (fft_mag_filtered_all[1]**2) / N**2
power_blue_filtered = (fft_mag_filtered_all[2]**2) / N**2

#---Before filtering--- (Weird that were using sum and mean, but maybe makes sense)
signal_power_red_before = np.sum(power_red_before[signal_indices_before])
signal_power_green_before = np.sum(power_green_before[signal_indices_before])
signal_power_blue_before = np.sum(power_blue_before[signal_indices_before])

noise_power_red_before = np.mean(np.delete(power_red_before, signal_indices_before))
noise_power_green_before = np.mean(np.delete(power_green_before, signal_indices_before))
noise_power_blue_before = np.mean(np.delete(power_blue_before, signal_indices_before))

SNR_red_before = 10 * np.log10(signal_power_red_before / noise_power_red_before)
SNR_green_before = 10 * np.log10(signal_power_green_before / noise_power_green_before)
SNR_blue_before = 10 * np.log10(signal_power_blue_before / noise_power_blue_before)

print("\nSNR før filtrering:")
print(f"Rød: {SNR_red_before:.2f} dB")
print(f"Grønn: {SNR_green_before:.2f} dB")
print(f"Blå: {SNR_blue_before:.2f} dB")

#---After filtering---
signal_power_red_filtered = np.sum(power_red_filtered[signal_indices_filtered])
signal_power_green_filtered = np.sum(power_green_filtered[signal_indices_filtered])
signal_power_blue_filtered = np.sum(power_blue_filtered[signal_indices_filtered])

noise_power_red_filtered = np.mean(np.delete(power_red_filtered, signal_indices_filtered))
noise_power_green_filtered = np.mean(np.delete(power_green_filtered, signal_indices_filtered))
noise_power_blue_filtered = np.mean(np.delete(power_blue_filtered, signal_indices_filtered))

SNR_red_filtered = 10 * np.log10(signal_power_red_filtered / noise_power_red_filtered)
SNR_green_filtered = 10 * np.log10(signal_power_green_filtered / noise_power_green_filtered)
SNR_blue_filtered = 10 * np.log10(signal_power_blue_filtered / noise_power_blue_filtered)

print("\nSNR etter filtrering:")
print(f"Rød: {SNR_red_filtered:.2f} dB")
print(f"Grønn: {SNR_green_filtered:.2f} dB")
print(f"Blå: {SNR_blue_filtered:.2f} dB")