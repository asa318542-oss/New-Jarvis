import sounddevice as sd
import numpy as np

# ===== MOSTRA DISPOSITIVI =====
print("\n--- Microfoni disponibili ---")
devices = sd.query_devices()

for i, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        print(f"[{i}] {dev['name']}")

# ===== SCEGLI MICROFONO =====
device_id = None  # metti numero se vuoi forzare microfono

# ===== CALLBACK AUDIO =====
def audio_callback(indata, frames, time, status):

    if status:
        print(status)

    # Peak volume
    peak = np.max(np.abs(indata))

    # RMS volume (più utile per STT)
    rms = np.sqrt(np.mean(indata**2))

    # Barra visiva
    bars = int(peak * 50)
    bar = "█" * bars

    print(f"Peak: {peak:.3f} | RMS: {rms:.3f} {bar}")

# ===== STREAM AUDIO =====
with sd.InputStream(
    device=device_id,
    channels=1,
    samplerate=16000,
    callback=audio_callback
):
    print("\n🎤 Parla nel microfono (CTRL+C per uscire)")
    while True:
        sd.sleep(100)
