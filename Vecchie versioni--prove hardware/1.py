import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import pvporcupine
import sounddevice as sd
from resemblyzer.demo_utils import plot_similarity_matrix, plot_histograms  # dai demo 1 e 5

# -----------------------------
# Configurazione wakeword (Porcupine)
# -----------------------------
WAKEWORD_PPN = "audio_data/my_wakeword.ppn"
LANG_PV = "audio_data/lang.pv"

porcupine = pvporcupine.create(
    keyword_paths=[WAKEWORD_PPN],
    library_path=None,
    model_path=LANG_PV
)

# -----------------------------
# Configurazione Resemblyzer (mono-user)
# -----------------------------
encoder = VoiceEncoder("cuda")  # modello GPU

# Utente autorizzato (file di riferimento della tua voce)
reference_file = Path("audio_data/my_voice.wav")
reference_wav = preprocess_wav(reference_file)
reference_embed = encoder.embed_utterance(reference_wav)

# Ground truth reali per fake detection
ground_truth_dir = Path("audio_data/ground_truth")
gt_files = list(ground_truth_dir.glob("*.wav"))
gt_wavs = [preprocess_wav(f) for f in gt_files]
gt_embeds = np.array([encoder.embed_utterance(w) for w in gt_wavs])

# Soglie
SIMILARITY_THRESHOLD = 0.85
FAKE_THRESHOLD = 0.85

# -----------------------------
# Funzioni Demo-style
# -----------------------------
def check_similarity_demo1(chunk_embed):
    """Demo 1 style: similarità con utente autorizzato"""
    score = np.inner(chunk_embed, reference_embed)
    return score

def check_fake_demo5(chunk_embed):
    """Demo 5 style: confronto con ground truth"""
    scores = np.inner(chunk_embed, gt_embeds)
    avg_score = scores.mean()
    return avg_score

def process_wakeword_chunk_demo(chunk_audio):
    """Unisce Demo 1 + Demo 5"""
    chunk_embed = encoder.embed_utterance(chunk_audio)

    # Demo 1: similarità voce
    similarity_score = check_similarity_demo1(chunk_embed)
    voice_ok = similarity_score > SIMILARITY_THRESHOLD

    # Demo 5: fake detection
    fake_score = check_fake_demo5(chunk_embed)
    fake_ok = fake_score > FAKE_THRESHOLD

    if voice_ok and fake_ok:
        print(f"[ACCEPTED] Wakeword riconosciuta! Similarity={similarity_score:.2f}, Fake={fake_score:.2f}")
        return True
    else:
        print(f"[REJECTED] Similarity={similarity_score:.2f}, Fake={fake_score:.2f}")
        return False

# -----------------------------
# Callback microfono
# -----------------------------
def audio_callback(indata, frames, time, status):
    pcm = indata[:, 0]  # mono
    if porcupine.process(pcm):
        print("Wakeword rilevata! Controllo voce e fake...")
        accepted = process_wakeword_chunk_demo(pcm)
        if accepted:
            print("Sistema attivato!")
        else:
            print("Voce non riconosciuta o fake")

# -----------------------------
# Avvio microfono
# -----------------------------
with sd.InputStream(
    channels=1,
    samplerate=porcupine.sample_rate,
    blocksize=porcupine.frame_length,
    callback=audio_callback
):
    print("Listening... premi Ctrl+C per terminare")
    while True:
        sd.sleep(1000)
