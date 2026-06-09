from wakewordFile import Wakeword
from res_whisp import run_whisper
import time
import ollama
import subprocess
from voice import speak
from sqlite import init_db,save,Load_History,Clean
from ddgs import DDGS
import socket

def start_ollama():
    try:
        import httpx
        httpx.get("http://localhost:11434")
    except:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
def has_internet(host="8.8.8.8",port=53,timeout=1)->bool:
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False
    
def web_search(query:str,max_results:int=3):
    try:
        with DDGS() as ddgs:
            results=list(ddgs.text(query,max_results=max_results))
            print(f"Risultati trovati: {len(results)}")  # ← debug
            
            if not results:
                return ""
            return "\n".join(f"-{r['title']}:{r['body'][:300]}" for r in results)
    except Exception as e:
        print(f"Errore ricerca: {e}")
        return ""# ← debug
def stream_risposta(response)-> str:
     full_text = ""
     buffer = ""
     for chunk in response:
            piece=chunk.message.content
            print(piece,end="",flush=True)
            full_text+=piece
            buffer+=piece
            while any(p in buffer for p in [".", "?", "!", ":"]):
             for p in [".","?","!",":"]:
               if p in buffer:
                parts=buffer.split(p,1)
                speak((parts[0]+p).strip())
                buffer=parts[1]
                break
     if buffer.strip():
             speak(buffer.strip())
     return full_text
     
    
def main():
    print("Sistema pronto...")
    start_ollama()
    conn = init_db()
    while True:
        
        print("Attendo wake word...")
        Wakeword() 
      

       
        print("Altri task possono partire qui...")
        user_text=run_whisper()
        if not user_text:
            continue
        while user_text:
                
            messages = Load_History(conn)
            current_message = {"role": "user", "content": user_text}

            if has_internet():
                print("searching online")
                risultati=web_search(user_text)
                if risultati:
                    current_message["content"]=(
                         f"{user_text}\n\n"
                        f"[Dati trovati online — usali solo se pertinenti]\n"
                        f"{risultati}"
                    )
            messages.append(current_message)

            save(conn, "user", user_text)

        
        
            response = ollama.chat(
                model="jarvis",
                messages=messages,
                stream=True
            )
            full_text=stream_risposta(response)
       
            save(conn, "assistant", full_text)
            Clean(conn)
            time.sleep(0.5)
            user_text=run_whisper(duration=5)
            if not user_text:
                print("secondo tentativo\n")
                user_text=run_whisper(duration=5)

if __name__ == "__main__":
    main()
