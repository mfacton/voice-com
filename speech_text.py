import sys
import os

import torch
from transformers import pipeline

import pyaudio
import wave
from pynput import keyboard

#sys.stderr = open(os.devnull, 'w')

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    torch_dtype=torch.float16,
    device="cuda",
    model_kwargs={"attn_implementation": "sdpa"},
)

audio_file_path = "input.wav"

channels = 1
rate = 44100
format = pyaudio.paInt16
chunk_size = 1024

audio = pyaudio.PyAudio()
stream = audio.open(format=format, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk_size)

print("Press SPACE to start/stop recording...")

frames = []
recording = False

def on_press(key):
    global recording, frames
    if key == keyboard.Key.space:  # Check if space is pressed
        if not recording:
            print("Recording started.")
            recording = True
        else:
            recording = False

listener = keyboard.Listener(on_press=on_press)
listener.start()

while not recording:
    pass

while recording:
    data = stream.read(chunk_size) 
    frames.append(data)

stream.stop_stream()
stream.close()
audio.terminate()

with wave.open(audio_file_path, 'wb') as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))

output = pipe(
    audio_file_path,
    chunk_length_s=30,
    batch_size=24,
    return_timestamps=False,
)
speech_text = output["text"]

print(speech_text)
