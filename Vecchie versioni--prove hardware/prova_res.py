import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import sounddevice as sd
from collections import deque
import queue
import threading

# --- Config ---
Base_Dir = Path(__file__).parent
Chunk_seconds = 5   # lunghezza del chunk in secondi
Sample = 16000      # sample rate
similarity_threshold = 0.70
sample = Sample * Chunk_seconds

# --- Inizializza VoiceEncoder ---
try:
    encoder = VoiceEncoder("cuda")
except:
    encoder = VoiceEncoder("cpu")

# --- Carica tutte le voci ground ---
ground_truth_dir = Base_Dir / "Voci" / "ground"
gt_f = list(ground_truth_dir.glob("*.wav"))
gt_names = [f.stem for f in gt_f]

gt_wavs = [preprocess_wav(f) for f in gt_f]
gt_emb = np.array([encoder.embed_utterance(w) for w in gt_wavs])
gt_emb = np.array([e / np.linalg.norm(e) for e in gt_emb])  # normalizzazione

# --- Funzione per verificare la voce ---
def verify_voice(chunk_audio):
    chunk_embed = encoder.embed_utterance(chunk_audio)
    chunk_embed = chunk_embed / np.linalg.norm(chunk_embed)

    sims = np.dot(gt_emb, chunk_embed)
    max_i = sims.argmax()
    max_s = sims[max_i]

    print(f"Max similarity: {max_s:.3f} con {gt_names[max_i]}")
    if max_s > similarity_threshold:
        print(f"✔ Voce riconosciuta: {gt_names[max_i]}")
    else:
        print("❌ Voce non riconosciuta")

# --- Buffer e coda per chunk ---
buffer = deque(maxlen=sample)
audio_queue = queue.Queue()

# --- Callback microfono ---
def audio_callback(indata, frames, time, status):
    buffer.extend(indata[:, 0])
    if len(buffer) >= sample:
        audio_np = np.array(buffer)
        buffer.clear()
        audio_queue.put(audio_np)

# --- Worker che processa i chunk ---
def worker():
    while True:
        audio_np = audio_queue.get()
        chunk_audio = preprocess_wav(audio_np)
        verify_voice(chunk_audio)
        audio_queue.task_done()

# --- Funzione principale ---
def run_realtime():
    threading.Thread(target=worker, daemon=True).start()
    with sd.InputStream(channels=1, samplerate=Sample, callback=audio_callback):
        print("🎤 Listening... parla ora!")
        while True:
            sd.sleep(1000)

# --- Entry point ---
if __name__ == "__prova_res__":
    run_realtime()
