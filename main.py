import numpy as np
import sounddevice as sd
import webrtcvad
import collections
from scipy.io.wavfile import write
import openai
from elevenlabs import generate, stream

SAMPLE_RATE = 16000  # VAD only supports 16kHz sample rate
FRAME_DURATION_MS = 30  # ms
BUFFER_DURATION_MS = 300  # ms
BUFFER_SIZE = int(SAMPLE_RATE * (BUFFER_DURATION_MS / 1000.0))
VAD_AGGRESSIVENESS = 3  # 0-3, 3 is the most aggressive
FILENAME = "temp.wav"
# Setup the Voice Activity Detector
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
ring_buffer = collections.deque(maxlen=BUFFER_SIZE)
triggered = False
audio_data = []

def callback(indata, frames, time, status):
    global triggered
    global audio_data

    # Voice detection
    is_speech = vad.is_speech(indata[:int(SAMPLE_RATE*0.03)].tobytes(), SAMPLE_RATE)

    if triggered:
        audio_data.append(indata.copy())
        num_unvoiced = len([f for f, speech in ring_buffer if not speech])
        if len(ring_buffer) == ring_buffer.maxlen and num_unvoiced > 0.9 * ring_buffer.maxlen:
            print('Finished recording!')
            write(FILENAME, SAMPLE_RATE, np.concatenate(audio_data, axis=0))  # Save as WAV file
            audio_file = open(FILENAME, "rb")
            transcript = openai.Audio.translate("whisper-1", audio_file)
            audio_stream = generate(
                text=transcript,
                stream=True
            )
            stream(audio_stream)
            # Clear the audio data and set triggered to False for next speech segment
            audio_data = []
            triggered = False
    else:
        ring_buffer.append((indata.copy(), is_speech))
        num_voiced = len([f for f, speech in ring_buffer if speech])
        if len(ring_buffer) == ring_buffer.maxlen and num_voiced > 0.5 * ring_buffer.maxlen:
            print('Starting to record!')
            triggered = True

with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE):
    while True:
        sd.sleep(1000)
