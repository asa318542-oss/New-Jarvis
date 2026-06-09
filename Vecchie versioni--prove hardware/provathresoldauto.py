import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import sounddevice as sd
import threading
import queue
from collections import deque
from faster_whisper import WhisperModel

# ================= CONFIG =================
Base_Dir = Path(__file__).parent
Chunk_seconds = 8        # durata chunk audio in secondi
Sample_rate = 16000      # sample rate
sample = Sample_rate * Chunk_seconds

# ================= WHISPER =================
try:
    Whisper = WhisperModel(
        "medium",
        device="cuda",
        compute_type="float16"
    )
except:
    Whisper = WhisperModel(
        "medium",
        device="cpu",
        compute_type="float16"
    )

# ================= ENCODER =================
try:
    encoder = VoiceEncoder("cuda")
except:
    encoder = VoiceEncoder("cpu")

# ================= GROUND TRUTH =================
ground_truth_dir = Base_Dir / "Voci" / "ground"
gt_files = list(ground_truth_dir.glob("*.wav"))
gt_names = [f.stem for f in gt_files]
gt_wavs = [preprocess_wav(f) for f in gt_files]
gt_emb = np.array([encoder.embed_utterance(w) for w in gt_wavs])

# normalizza embeddings
def normalize(v):
    return v / np.linalg.norm(v)

gt_emb = np.array([normalize(e) for e in gt_emb])

# ================= THRESHOLD AUTOMATICO =================
def compute_threshold(gt_emb):
    n = len(gt_emb)
    max_sims = []
    for i in range(n):
        others = np.delete(gt_emb, i, axis=0)
        sims = np.dot(others, gt_emb[i])
        max_sims.append(np.max(sims))
    max_sims = np.array(max_sims)
    print("Max similarity tra ground truth diversi:", max_sims)
    threshold = np.max(max_sims) + 0.05  # margine 0.05 per sicurezza
    threshold = min(threshold, 0.99)
    print(f"Threshold automatico calcolato: {threshold:.3f}")
    return threshold

similarity_threshold = compute_threshold(gt_emb)

# ================= AUDIO BUFFER =================
buffer = deque(maxlen=sample)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    global buffer
    buffer.extend(indata[:,0])
    if len(buffer) >= sample:
        audio_np = np.array(buffer)
        buffer.clear()
        audio_queue.put(audio_np)

# ================= FUNZIONE VERIFICA VOCE =================
def verify_voice(chunk_audio):
    # normalizza audio
    chunk_audio = chunk_audio / np.max(np.abs(chunk_audio))
    
    # estrai embedding
    chunk_embed = encoder.embed_utterance(chunk_audio)
    chunk_embed = normalize(chunk_embed)
    
    # calcola similarità
    sims = np.dot(gt_emb, chunk_embed)
    
    # debug: stampa tutte le similarità
    for name, sim in zip(gt_names, sims):
        print(f"{name}: {sim:.3f}")
    
    max_i = sims.argmax()
    max_s = sims[max_i]
    print(f"Max similarity: {max_s:.3f} con {gt_names[max_i]}")

    return max_s > similarity_threshold, gt_names[max_i]

# ================= WORKER =================
def worker():
    while True:
        audio_np = audio_queue.get()
        audio_pc = preprocess_wav(audio_np)
        accepted, name = verify_voice(audio_pc)
        if accepted:
            print(f"✔ Utente Riconosciuto: {name}")
            segments, _ = Whisper.transcribe(
                audio_np.astype(np.float32),
                language="it"
            )
            text = "".join([seg.text for seg in segments])
            print(f"📄 Trascrizione: {text}")
        else:
            print("❌ Utente non riconosciuto")
        audio_queue.task_done()

# ================= MAIN =================
def run_whisper():
    threading.Thread(target=worker, daemon=True).start()
    with sd.InputStream(
        channels=1,
        samplerate=Sample_rate,
        callback=audio_callback
    ):
        print("🎤 Listening... parla ora!")
        while True:
            sd.sleep(1000)

# ================= ENTRY POINT =================
if __name__ == "__main__":
    run_whisper()
