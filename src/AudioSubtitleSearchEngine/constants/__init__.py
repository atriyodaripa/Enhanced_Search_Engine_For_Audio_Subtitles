from pathlib import Path

# CONFIG_FILE_PATH = Path("config/config.yaml")
# PARAMS_FILE_PATH = Path("params.yaml")

AUDIO_DIRECTURY = 'artifacts\\query\\audio\\'
TEXT_DIRECTORY = 'artifacts\\query\\text\\'
AUDIO_FILENAME = 'query_audio.wav'
TEXT_FILENAME = 'query_text.txt'

DB_DIRECTORY = 'artifacts\\input_data\\eng_subtitles_database.db'
DECODED_SUBTITLES_DIRECTORY = 'artifacts\\decoded_input'

CLEANED_SUBTITLES_DIRECTORY = 'artifacts\\cleaned_data\\'

MODEL_NAME = 'all-MiniLM-L6-v2'
VECTOR_DB_PATH = 'artifacts\\embedded_data\\vector_db\\db_embedding'


AUDIO_SAMPLERATE = 48000
AUDIO_DURATION = 30 # seconds