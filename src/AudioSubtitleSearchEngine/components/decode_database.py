import sqlite3
import pandas as pd
import zipfile
import io
from AudioSubtitleSearchEngine.constants import *
from AudioSubtitleSearchEngine.logging import logger


class DatabaseDecoder:
    def __init__(self, db_path=DB_DIRECTORY):
        self.conn = sqlite3.connect(db_path)
        self.df = pd.read_sql_query("SELECT * FROM zipfiles", self.conn)
        self.conn.close()
    
    def decode_subtitle_content(self, data):
        with io.BytesIO(data) as f:
            with zipfile.ZipFile(f, 'r') as zip_file:
                decompressed_subtitle_content = zip_file.read(zip_file.namelist()[0])
                decoded_subtitle_content = decompressed_subtitle_content.decode('latin-1')
        return decoded_subtitle_content
    
    def decode_and_save_subtitles(self, filedir=DECODED_SUBTITLES_DIRECTORY):
        self.df['decoded_content'] = self.df['content'].apply(self.decode_subtitle_content)
        # decoded_content = []
        # for index, row in self.df.iterrows():
        #     logger.info(f"contents are {row['content']}")
            # decoded_content.append(self.decode_subtitle_content(row['content']))
        # self.df['decoded_content'] = decoded_content
        
        self.df.to_pickle(f"{filedir}/subtitles_decoded.pkl")
        return f"{filedir}/subtitles_decoded.pkl"