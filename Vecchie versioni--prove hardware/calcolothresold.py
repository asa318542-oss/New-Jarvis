import numpy as np
import sounddevice as sd
from resemblyzer import VoiceEncoder, preprocess_wav
import itertools

SAMPLE_RATE = 16000
DURATION = 6
NUM_SAMPLES = SAMPLE_RATE * DURATION

print("Caricamento encoder...")
try:
    encoder = VoiceEncoder("cuda")
except:
    encoder = VoiceEncoder("cpu")

def normalize(v):
    return v / np.linalg.norm(v)

def record_audio(label):
    print(f"\n>>> {label}: parla per {DURATION} secondi...")
    audio = sd.rec(NUM_SAMPLES, samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio = audio.flatten()
    audio = preprocess_wav(audio)
    embed = encoder.embed_utterance(audio)
    return normalize(embed)

# ========= RACCOLTA DATI =========

print("\n=== REGISTRAZIONI UTENTE AUTORIZZATO ===")
user_embeddings = [record_audio(f"UTENTE {i+1}") for i in range(5)]

print("\n=== REGISTRAZIONI ALTRA PERSONA ===")
other_embeddings = [record_audio(f"ALTRO {i+1}") for i in range(5)]

# ========= CALCOLO SIMILARITÀ =========

# Genuine scores (tu contro tu)
genuine_scores = []
for a, b in itertools.combinations(user_embeddings, 2):
    genuine_scores.append(np.dot(a, b))

# Impostor scores (tu contro altro)
impostor_scores = []
for u in user_embeddings:
    for o in other_embeddings:
        impostor_scores.append(np.dot(u, o))

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print("\n--- STATISTICHE ---")
print(f"Genuine mean: {genuine_scores.mean():.3f}")
print(f"Impostor mean: {impostor_scores.mean():.3f}")

# ========= CALCOLO EER =========

all_scores = np.concatenate([genuine_scores, impostor_scores])
labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])

thresholds = np.linspace(min(all_scores), max(all_scores), 500)
best_threshold = None
min_diff = 1

for t in thresholds:
    false_accept = np.mean(impostor_scores >= t)
    false_reject = np.mean(genuine_scores < t)
    diff = abs(false_accept - false_reject)
    if diff < min_diff:
        min_diff = diff
        best_threshold = t
        eer = (false_accept + false_reject) / 2

print("\n=== RISULTATO FINALE ===")
print(f"Threshold ottimale (EER): {best_threshold:.3f}")
print(f"EER stimato: {eer:.3f}")

print("\nUsa questo valore nel tuo sistema.")