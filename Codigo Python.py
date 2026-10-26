import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]

# --- 1. Definición de la señal en el dominio del tiempo ---
fs = 1000  # Frecuencia de muestreo (Hz)
t = np.arange(0, 1.0, 1/fs)  # Vector de tiempo de 1 segundo

# Señal senoidal compuesta: 50 Hz y una componente de alta frecuencia en 120 Hz
f1, f2 = 50, 120
signal = 0.6 * np.sin(2 * np.pi * f1 * t) + 0.4 * np.sin(2 * np.pi * f2 * t)

# Pulso rectangular (Ventana)
pulse = np.zeros_like(t)
pulse[(t >= 0.3) & (t <= 0.7)] = 1.0

# --- 2. Cálculo de la Transformada de Fourier (np.fft.fft) ---
N = len(signal)
fft_signal = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(N, 1/fs)

# Tomamos solo la mitad positiva del espectro
positive_freqs = fft_freqs[:N//2]
magnitude = (2/N) * np.abs(fft_signal[:N//2])
phase = np.angle(fft_signal[:N//2])

# --- 3. Visualización de los Resultados ---
plt.figure(figsize=(12, 10))

# Gráfica en el dominio del tiempo (Señal Senoidal)
plt.subplot(3, 2, 1)
plt.plot(t, signal, color='b')
plt.title('Señal en el Dominio del Tiempo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)

# Espectro de Magnitud
plt.subplot(3, 2, 2)
plt.plot(positive_freqs, magnitude, color='r')
plt.title('Espectro de Frecuencia (Magnitud)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X(f)|')
plt.grid(True)

# Gráfica del Pulso Rectangular en el tiempo
plt.subplot(3, 2, 3)
plt.plot(t, pulse, color='g')
plt.title('Pulso Rectangular en el Tiempo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)

# Espectro de Fase
plt.subplot(3, 2, 4)
plt.plot(positive_freqs, phase, color='m')
plt.title('Espectro de Frecuencia (Fase)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True)

# Verificación de Propiedad de Linealidad
# a*x1(t) + b*x2(t) <--> a*X1(f) + b*X2(f)
signal_lin = 2 * np.sin(2 * np.pi * 50 * t) + 3 * np.sin(2 * np.pi * 120 * t)
fft_lin = np.fft.fft(signal_lin)
magnitude_lin = (2/N) * np.abs(fft_lin[:N//2])

plt.subplot(3, 2, 5)
plt.plot(positive_freqs, magnitude_lin, color='darkorange')
plt.title('Verificación Propiedad de Linealidad')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|X(f)|')
plt.grid(True)

plt.tight_layout()
plt.show()