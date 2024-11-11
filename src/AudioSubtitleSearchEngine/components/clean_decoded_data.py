import pandas as pd
import re
import math
from AudioSubtitleSearchEngine.constants import *
from AudioSubtitleSearchEngine.logging import logger

class DataCleaner:
    def __init__(self, data_path):
        self.df = pd.read_pickle(data_path)
    
    def data_cleaning(self, data):
        data = re.sub("\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}","",data)      # remove timestamps
        data = re.sub("\n*\d+\r","",data)       #remove integer characters present before each timestamp line
        data = re.sub("<i>|</i>", "", data)     #remove <i> and </i> tags
        data = re.sub("\r|\n", "", data)     # remove whitespaces and line breaks
        data = re.sub(r'[^\x00-\x7F]', '', data)      # remove non-ascii characters
        data = re.sub(r'www\.osdb\.link\/[\w\d]+\s+|www\.OpenSubtitles\.org|osdb\.link\/ext|api\.OpenSubtitles\.org|OpenSubtitles\.com', '', data)      # remove links
        
        return data.lower()     # lowering each chanracters
    
    def clean_and_save_data(self, output_path=CLEANED_SUBTITLES_DIRECTORY):
        logger.info("Inside clean_and_save_data function")
        self.df['cleaned_data'] = self.df['decoded_content'].apply(self.data_cleaning)
        self.df.drop(['content','num','decoded_content'], axis=1, inplace=True)
        logger.info("Cleaned the decoded data")

        index=[i for i in range(len(self.df))]
        self.df['index']=index
        logger.info(f"Cleaned dataframe: \n {self.df.head()}")

        num_row = 1000
        num_df = math.ceil(len(self.df)/num_row)

        segregated_df_path = []

        for i in range(num_df):
            start_idx = 0 + i*num_row
            end_idx = min(0 + (i + 1) * num_row, len(self.df))
            new_df = self.df[start_idx:end_idx]
            logger.info(f"Writing dataframe to {output_path}df_{i}.pkl")
            segregated_df_path.append(f"{output_path}df_{i}.pkl")
            new_df.to_pickle(f'{output_path}df_{i}.pkl')
        
        return segregated_df_path
        

