import sounddevice as sd
import soundfile as sf
import IPython.display as ipd
import streamlit as st
import librosa
import torch
from audiorecorder import audiorecorder
# from audio_recorder_streamlit import audio_recorder

from AudioSubtitleSearchEngine.constants import *
from AudioSubtitleSearchEngine.logging import logger
from AudioSubtitleSearchEngine.components.decode_database import DatabaseDecoder
from AudioSubtitleSearchEngine.components.clean_decoded_data import DataCleaner
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, AutoModelForCTC

class ProcessQuery(DatabaseDecoder, DataCleaner):
    def __init__(
        self,
        audio_directury = AUDIO_DIRECTURY,
        text_directory = TEXT_DIRECTORY,
        audio_filename = AUDIO_FILENAME,
        text_filename = TEXT_FILENAME,
        audio_samplerate = AUDIO_SAMPLERATE,
        audio_duration = AUDIO_DURATION):

        self.audio_directury = audio_directury
        self.text_directory = text_directory
        self.audio_filename = audio_filename
        self.text_filename = text_filename
        self.audio_samplerate = audio_samplerate
        self.audio_duration = audio_duration

        super().__init__()

    def record_audio(self):
        """
        Record audio for the given duration and save it in the specified directory.
        """
        self.audio_path = self.audio_directury + self.audio_filename
        logger.info("****** START RECORDING AUDIO ******")
        st.title("Audio Recorder")
        # audio_bytes = audio_recorder()
        # if audio_bytes:
        #     st.audio(audio_bytes, format="audio/wav")
        audio = audiorecorder("Click to record", "Click to stop recording")
        if len(audio) > 0:
            # To play audio in frontend:
            st.audio(audio.export().read())
            # logger.info("abcd")

            # To save audio to a file, use pydub export method:
            audio.export(self.audio_path, format="wav")
            # logger.info("efgh")

            # To get audio properties, use pydub AudioSegment properties:
            st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
            # logger.info("ijkl")

            # st.button("Search")


        # input_query_recording = sd.rec(int(self.audio_samplerate * self.audio_duration), samplerate=self.audio_samplerate,
        #     channels=1, blocking=True)
        # logger.info("****** END OF RECORDING ******")
        # sd.wait()
        # sf.write(self.audio_path, input_query_recording, self.audio_samplerate)
        return self.audio_path


    def convert_audio_to_text(self, audio_path):
        self.text_path = self.text_directory + self.text_filename
        samples, sample_rate = librosa.load(audio_path , sr = 16000)
        ipd.Audio(samples,rate=16000)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        input_values = self.processor(samples, return_tensors="pt", padding="longest").input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        with open(self.text_path, "w") as f:
            f.write(transcription[0])
        return self.text_path



    
