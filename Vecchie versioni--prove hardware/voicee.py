import subprocess

def speak(text):
    text = text.replace("'", "''")  # escape apostrofi
    script = f"""
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SelectVoice('Microsoft Cosimo Desktop')
    $synth.Rate = -2
    $synth.Speak('{text}')
    """
    subprocess.run(["powershell", "-Command", script], check=True)
