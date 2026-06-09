from pathlib import Path
import sounddevice as sd
import numpy as np

Base_dir = Path(__file__).parent

WAKEWORD = Base_dir / "wakeword" / "Hey-Jarvis_it_windows_v4_0_0.ppn"
LANGUAGE = Base_dir / "wakeword" / "porcupine_params_it.pv"
ACCESS_KEY = ""
ONNX_PATH = Base_dir / "wakeword" / "hey_jar_viss.onnx"
THRESHOLD = 0.5
USA_PORCUPINE = False

try:
    import pvporcupine
    porcupine = pvporcupine.create(
        keyword_paths=[str(WAKEWORD)],
        library_path=None,
        model_path=str(LANGUAGE),
        access_key=ACCESS_KEY,
        sensitivities=[0.8]
    )
    USA_PORCUPINE = True
    print("Porcupine caricato")

except:
    from openwakeword.model import Model
    modello_onnx = Model(wakeword_models=[str(ONNX_PATH)], inference_framework="onnx")
    USA_PORCUPINE = False
    print("openWakeWord caricato")


def Wakeword():
    print("In ascolto...")

    detected = False

    if USA_PORCUPINE:

        def audio_callback(indata, frames, time, status):
            nonlocal detected
            pcm = (indata[:, 0] * 32767).astype(np.int16)
            if porcupine.process(pcm) >= 0:
                print("Rilevata con Porcupine!")
                detected = True
                raise sd.CallbackStop()

        with sd.InputStream(
            channels=1,
            samplerate=porcupine.sample_rate,
            callback=audio_callback,
            blocksize=porcupine.frame_length
        ):
            while not detected:
                sd.sleep(50)

    else:

        def audio_callback(indata, frames, time, status):
            nonlocal detected
            pcm = (indata[:, 0] * 32767).astype(np.int16)
            prediction = modello_onnx.predict(pcm)
            for score in prediction.values():
                if score >= THRESHOLD:
                    print("Rilevata con openWakeWord!")
                    detected = True
                    raise sd.CallbackStop()

        with sd.InputStream(
            channels=1,
            samplerate=16000,
            callback=audio_callback,
            blocksize=1280
        ):
            while not detected:
                sd.sleep(50)
        
        modello_onnx.reset()