import threading, queue, wave, subprocess, sys, os
import numpy as np
import sounddevice as sd

PIPER_CMD = r"C:\Users\panes\AppData\Local\Programs\Python\Python311\Scripts\piper.exe"
ONNX_PATH = r"C:\Users\panes\Desktop\piper_voci_italiano\Leonardo\leonardo-epoch=2024-step=996300.onnx"
JSON_PATH = r"C:\Users\panes\Desktop\piper_voci_italiano\Leonardo\leonardo-epoch=2024-step=996300.json"

# Assicura che esista il .onnx.json
json_atteso = ONNX_PATH + ".json"
if not os.path.exists(json_atteso):
    import shutil
    shutil.copy2(JSON_PATH, json_atteso)

speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break

        tmp = os.path.join(os.path.dirname(ONNX_PATH), "_tmp_out.wav")
        subprocess.run(
            [PIPER_CMD, "--model", ONNX_PATH, "--output_file", tmp],
            input=text.encode("utf-8"),
            capture_output=True,
        )
        with wave.open(tmp, "rb") as wav:
            rate  = wav.getframerate()
            raw   = wav.readframes(wav.getnframes())
        os.unlink(tmp)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio) > 0:
            sd.play(audio, samplerate=rate)
            sd.wait()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)