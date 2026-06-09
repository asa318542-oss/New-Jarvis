import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import sounddevice as sd
import threading
import queue
from collections import deque
from faster_whisper import WhisperModel


Base_Dir=Path(__file__).parent



Chunk_seconds=7
Sample=16000
similarity=0.600

sample=Sample*Chunk_seconds

try:
    Whisper=WhisperModel(
        "medium",
        device="cuda",
        compute_type="float16"
    )
    
    
except:
    Whisper=WhisperModel(
        "small",
        device="cpu",
        compute_type="float16"
    )
    
    
buffer=deque(maxlen=sample)

try:
    encoder=VoiceEncoder("cuda")
except:
    encoder=VoiceEncoder("cpu")
    

ground_truth_dir=Base_Dir/"Voci"/"ground"
gt_f=list(ground_truth_dir.glob("*.wav"))
gt_names=[f.stem for f in gt_f]
gt_wavs=[preprocess_wav(f) for f in gt_f]
gt_emb=np.array([encoder.embed_utterance(w) for w in gt_wavs])

def normalize(v):
    return v/np.linalg.norm(v)


gt_emb=np.array([normalize(e) for e in gt_emb])


def demo15(chunk_audio):
    max_val=np.max(np.abs(chunk_audio))
    if max_val>0:
        chunk_audio=chunk_audio/max_val
    else:
        print("audio vuoto")
        return False,None
    chunk_embed=encoder.embed_utterance(chunk_audio)
    chunk_embed=normalize(chunk_embed)
    
    sims=np.dot(gt_emb,chunk_embed)
    max_i=sims.argmax()
    max_s=sims[max_i]
    print(f"Max similarity: {max_s:.3f} con {gt_names[max_i]}")
    return max_s>similarity,gt_names[max_i]

    
    
    
audio_queue=queue.Queue()
text_queue=queue.Queue()   
def audio_callback(indata,frames,time,status):
    global buffer
    
    buffer.extend(indata[:,0])
    # Se siamo in registrazione, accumulo audio
    
    if len(buffer)>=sample:
            audio=np.array(buffer)
            buffer.clear()

            audio_queue.put(audio)
    
       
          
    
def worker():
    while True:
            audio_np=audio_queue.get()
            audio_pc=preprocess_wav(audio_np) 
                        # Speaker verification
            accepted,name=demo15(audio_pc)
            if accepted and name is not None:
                print("Utente Riconosciuto")
                segments,_=Whisper.transcribe(
                    audio_pc.astype(np.float32),
                    language="it",
                    beam_size=10,
                    temperature=0.1
                )
                if not segments:
                    print("stringa vuota")
                else:
                    text="".join([seg.text for seg in segments])
                    print(text)
                    text_queue.put(text)
            else:
                print("Utente non riconosciuto")
                
            audio_queue.task_done()
            
           
def run_whisper(duration=None):
    global Chunk_seconds, sample, buffer
    if duration is not None:
        Chunk_seconds = duration
        sample = Sample * Chunk_seconds
        buffer = deque(maxlen=sample)
        
    threading.Thread(target=worker,daemon=True).start()

    with sd.InputStream(
          channels=1,
          samplerate=Sample,
          callback=audio_callback
      ):    
       print("listening")
       try:
        text=text_queue.get(timeout=duration)
        return text
       except queue.Empty:
           return None

