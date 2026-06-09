# 🤖 New Jarvis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLM-black?logo=ollama&logoColor=white)
![Piper](https://img.shields.io/badge/Piper-TTS%20Italiano-green)
![SQLite](https://img.shields.io/badge/SQLite-Memoria-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

> Un assistente vocale personale locale, ispirato a Jarvis. Si attiva con una wake word, riconosce la tua voce, ragiona con un LLM in locale e ti risponde parlando — tutto offline, tutto tuo.

---

## 📋 Indice

- [Descrizione](#-descrizione)
- [Funzionalità](#-funzionalità)
- [Struttura del progetto](#-struttura-del-progetto)
- [Requisiti](#-requisiti)
- [Installazione](#-installazione)
- [Configurazione](#-configurazione)
- [Utilizzo](#-utilizzo)

---

## 📖 Descrizione

**New Jarvis** è un assistente vocale completamente locale che combina più tecnologie open source per offrire un'esperienza simile a un assistente AI personale:

| Componente | Tecnologia | Ruolo |
|---|---|---|
| 🎤 Wake Word | `wakewordFile.py` + modelli custom | Attiva Jarvis quando lo chiami |
| 🎙️ Speech-to-Text | [OpenAI Whisper](https://github.com/openai/whisper) | Trascrive l'audio in testo |
| 🧠 LLM | [Ollama](https://ollama.com/) | Elabora e genera la risposta |
| 🔊 Text-to-Speech | [Piper](https://github.com/rhasspy/piper) (voci italiano) | Risponde con voce italiana naturale |
| 🧬 Speaker ID | [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) | Riconosce chi sta parlando |
| 🧠 Memoria | SQLite (`memory.db`) | Ricorda le conversazioni passate |

Nessuna API esterna. Nessun dato inviato al cloud. Gira tutto sulla tua macchina.

---

## ✨ Funzionalità

- 💤 **Wake word** personalizzata: Jarvis si attiva solo quando lo chiami
- 🎤 **Riconoscimento vocale** preciso tramite Whisper
- 🧬 **Riconoscimento speaker** tramite Resemblyzer (sa chi parla)
- 🧠 **LLM locale** tramite Ollama (es. `llama3`, `mistral`, `gemma`)
- 🔊 **Voce italiana** naturale con Piper TTS
- 💾 **Memoria persistente** delle conversazioni su SQLite
- ⚡ Funzionamento completamente **offline**

---

## 📁 Struttura del progetto

```
new-jarvis/
│
├── Jarvis.py               # Entry point — orchestra tutti i moduli
├── res_whisp.py            # Modulo Whisper: speech-to-text
├── voice.py                # Modulo TTS: sintesi vocale con Piper
├── wakewordFile.py         # Rilevamento wake word
├── sqlite.py               # Gestione memoria conversazioni (SQLite)
│
├── memory.db               # Database SQLite con la cronologia
│
├── ollama/                 # Configurazione e modelli Ollama
├── piper_voci_italiano/    # Modelli voce Piper in italiano
├── resemblyzer/            # Modelli e config per il riconoscimento speaker
├── wakeword/               # Modelli per la wake word
├── Voci/                   # File audio / voci aggiuntive
│
├── venv_gpu/               # Ambiente virtuale (GPU)
├── Vecchie versioni--prove hardware/  # Archivio versioni precedenti
└── __pycache__/
```

---

## 🛠️ Requisiti

- Python **3.10+**
- [Ollama](https://ollama.com/) installato e in esecuzione
- `ffmpeg` installato nel sistema
- Un microfono funzionante
- **GPU Nvidia consigliata** (CUDA) per Whisper in tempo reale

---

## 📦 Installazione

### 1. Clona il repository

```bash
git clone https://github.com/tuo-username/new-jarvis.git
cd new-jarvis
```

### 2. Crea un ambiente virtuale

```bash
python -m venv venv_gpu
# Linux / macOS
source venv_gpu/bin/activate
# Windows
venv_gpu\Scripts\activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Installa un modello Ollama

```bash
ollama pull llama3
```

### 5. Scarica una voce Piper italiana

Inserisci i file `.onnx` e `.json` della voce scelta nella cartella `piper_voci_italiano/`.
Le voci sono disponibili su [huggingface.co/rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).

---

## ⚙️ Configurazione

Crea un file `.env` nella root del progetto:

```bash
cp .env.example .env
```

Modifica il file `.env` con le tue preferenze:

```env
# ── Ollama ──────────────────────────────────────────
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://localhost:11434

# ── Whisper ─────────────────────────────────────────
# Modelli: tiny | base | small | medium | large
WHISPER_MODEL=base
WHISPER_LANGUAGE=it

# ── Piper TTS ───────────────────────────────────────
# Nome del file .onnx nella cartella piper_voci_italiano/
PIPER_VOICE=it_IT-paola-medium.onnx

# ── Wake Word ───────────────────────────────────────
WAKEWORD_MODEL=wakeword/jarvis.ppn
WAKEWORD_SENSITIVITY=0.5

# ── Memoria SQLite ──────────────────────────────────
DB_PATH=memory.db
# Numero di messaggi passati da includere nel contesto
MEMORY_CONTEXT_LIMIT=10

# ── Resemblyzer ─────────────────────────────────────
SPEAKER_RECOGNITION=true
```

---

## 🚀 Utilizzo

Assicurati che Ollama sia in esecuzione:

```bash
ollama serve
```

Avvia Jarvis:

```bash
python Jarvis.py
```

Jarvis si metterà in ascolto della **wake word**. Quando la sente, inizia a registrare il tuo comando, lo elabora e risponde a voce.

Premi `Ctrl+C` per uscire.
