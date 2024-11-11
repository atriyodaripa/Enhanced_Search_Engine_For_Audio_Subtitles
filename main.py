from AudioSubtitleSearchEngine.logging import logger
from AudioSubtitleSearchEngine.components.query_processing import ProcessQuery
from AudioSubtitleSearchEngine.components.decode_database import DatabaseDecoder
from AudioSubtitleSearchEngine.components.clean_decoded_data import DataCleaner
from AudioSubtitleSearchEngine.components.doc_chunk_embedding import DocChunker
from AudioSubtitleSearchEngine.constants import *
import re
import chromadb
from sentence_transformers import SentenceTransformer
import streamlit as st
import IPython.display as ipd


import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoProcessor, AutoModelForCTC
import torch

logger.info("Welcome to custom logging.")

st.header("Movie Subtitle Search Engine")
# Process audio query and convert it to text
proc_query = ProcessQuery()
recording_path = proc_query.record_audio()
text_path = proc_query.convert_audio_to_text(recording_path)
logger.info(text_path)


# # database_decoder = DatabaseDecoder()
# # decoded_subtitles = database_decoder.decode_and_save_subtitles()

# # decoded_subtitles = 'artifacts\\decoded_input\\subtitles_decoded.pkl'
# # data_cleaner = DataCleaner(decoded_subtitles)
# # segregated_df_path_list = data_cleaner.clean_and_save_data()

# # chunk_embedder = DocChunker(segregated_df_path_list)
# # doc_chunks = chunk_embedder.doc_chunking()
# # chunk_embeddings = chunk_embedder.doc_embedding(doc_chunks)
# # chunk_embedder.save_embedding(chunk_embeddings)

def extract_id(id_list):
    new_id_list=[]
    for item in id_list:
        match = re.match(r'^(\d+)', item)
        if match:
            extracted_number = match.group(1)
            new_id_list.append(extracted_number)
    return new_id_list


chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
query_collection = chroma_client.get_collection(name="eng_subtitles")

if st.button("Search")==True:

    st.subheader("Relevant Subtitle Files")
    with open(text_path, 'r') as file:
        text = file.read()
    search_query_text=proc_query.data_cleaning(text)
    logger.info(f"search query text: {search_query_text}")
    # query_embed = model.encode(search_query).tolist()

    search_results=query_collection.query(query_texts=[search_query_text], n_results=3)
    logger.info(f"search_results metadatas: {search_results['metadatas']}")
    logger.info(f"search_results id_list: {search_results['ids']}")

    meta_datas = search_results['metadatas'][0]
    id_list = search_results['ids'][0]

    id_list = extract_id(id_list)
    print(id_list)
    for id in id_list:
        file_name = query_collection.get(ids=f"{id}")["metadatas"][0]
        st.markdown(f"[{file_name}](https://www.opensubtitles.org/en/subtitles/{id})")

