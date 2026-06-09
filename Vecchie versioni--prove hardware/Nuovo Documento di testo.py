#!/usr/bin/env python3
"""
Scarica le voci Piper TTS italiano da HuggingFace
Repository: kirys79/piper_italiano
Voci disponibili: Aurora, Leonardo, Giorgio (fine-tune di Leonardo)
"""

import os
import sys
import subprocess

# ── Installa huggingface_hub se non presente ──────────────────────────────────
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("📦 Installo huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download

# ── Configurazione ────────────────────────────────────────────────────────────

REPO_ID      = "kirys79/piper_italiano"
VOCI         = ["Aurora", "Leonardo", "Giorgio"]
CARTELLA_OUT = os.path.join(os.path.expanduser("~"), "piper_voci_italiano")

# ── Selezione voci ────────────────────────────────────────────────────────────

print("\n🎙️  Downloader voci Piper TTS - Italiano")
print("=" * 45)
print("Voci disponibili:")
print("  1) Aurora   – voce femminile (from scratch)")
print("  2) Leonardo – voce maschile (from scratch)")
print("  3) Giorgio  – voce maschile (fine-tune di Leonardo)")
print("  4) Tutte    – scarica tutto (~2.73 GB)")
print()

scelta = input("Quale voce vuoi scaricare? [1/2/3/4]: ").strip()

if scelta == "1":
    da_scaricare = ["Aurora"]
elif scelta == "2":
    da_scaricare = ["Leonardo"]
elif scelta == "3":
    da_scaricare = ["Giorgio"]
elif scelta == "4":
    da_scaricare = VOCI
else:
    print("⚠️  Scelta non valida. Scarico tutte le voci.")
    da_scaricare = VOCI

# ── Download ──────────────────────────────────────────────────────────────────

print(f"\n📁 I file verranno salvati in: {CARTELLA_OUT}")
print(f"⬇️  Scarico: {', '.join(da_scaricare)}\n")

os.makedirs(CARTELLA_OUT, exist_ok=True)

# Costruisce i pattern per scaricare solo le cartelle selezionate
# Include anche i file root (.gitattributes, README.md)
patterns = [f"{voce}/*" for voce in da_scaricare] + ["README.md", ".gitattributes"]

try:
    percorso = snapshot_download(
        repo_id=REPO_ID,
        local_dir=CARTELLA_OUT,
        allow_patterns=patterns,
        ignore_patterns=["*.csv", "wavs/*"],  # salta dataset audio di training
        repo_type="model",
    )
    print(f"\n✅ Download completato!")
    print(f"📂 File salvati in: {percorso}\n")

    # Mostra i file scaricati
    print("File scaricati:")
    for radice, cartelle, files in os.walk(percorso):
        livello = radice.replace(percorso, "").count(os.sep)
        indent  = "  " * livello
        cartella_nome = os.path.basename(radice)
        if livello > 0:
            print(f"{indent}📁 {cartella_nome}/")
        for f in files:
            sotto = "  " * (livello + 1)
            percorso_f = os.path.join(radice, f)
            dim = os.path.getsize(percorso_f)
            dim_str = f"{dim / 1024 / 1024:.1f} MB" if dim > 1024*1024 else f"{dim / 1024:.1f} KB"
            print(f"{sotto}📄 {f}  ({dim_str})")

    print("\n💡 Come usare le voci con Piper TTS:")
    print("   piper --model <percorso/voce.onnx> --output_file output.wav <<< 'Testo da sintetizzare'")
    print("   Esempio:")
    if "Leonardo" in da_scaricare:
        print(f"   piper --model {CARTELLA_OUT}/Leonardo/<file.onnx> --output_file voce.wav <<< 'Ciao mondo'")

except Exception as e:
    print(f"\n❌ Errore durante il download: {e}")
    print("   Verifica la connessione internet e riprova.")
    sys.exit(1)