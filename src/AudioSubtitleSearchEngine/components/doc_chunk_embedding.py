import pandas as pd
import torch
import chromadb
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from AudioSubtitleSearchEngine.logging import logger
from AudioSubtitleSearchEngine.constants import *

class DocChunker:
    def __init__(self, df_path_list, model_name = MODEL_NAME, vector_db_path = VECTOR_DB_PATH):
        self.df_path_list = df_path_list
        self.text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
                                                       chunk_size=200,
                                                       chunk_overlap=50
                                                       )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        
    def doc_chunking(self):
        num_df = len(self.df_path_list)
        for i in range(num_df):
            df_loc = self.df_path_list[i]
            df = pd.read_pickle(df_loc)
            loader = DataFrameLoader(df, page_content_column='decoded_content')
            data  = loader.load()
            doc_contents = [doc.page_content for doc in data]
            meta_datas = [doc.metadata for doc in data]
            logger.info(f"********* Started document chunker for {i} dataframe *********")
            self.chunks = self.text_splitter.create_documents(doc_contents, meta_datas)
            logger.info(f"********* Finished Document Chunking for {i} dataframe *********")
        return self.chunks
        
    def doc_embedding(self, doc_chunks):
        num_df = len(self.df_path_list)
        for i in range(num_df):
            logger.info(f"********* Started embedding the document chunks for {i} dataframe *********")
            self.embedded_chunks = self.model.encode([chunk.page_content for chunk in doc_chunks]).tolist()
            logger.info(f"********* Finished embedding the document chunks for {i} dataframe *********")
            logger.info("******************"*10)
        return self.embedded_chunks

    def save_embedding(self, embedded_chunks):
        collection = self.chroma_client.get_or_create_collection(name="eng_subtitles", metadata={"hnsw:space": "cosine"})
        logger.info(f"********* Started storing the embeddings to chroma db for {i} dataframe *********")
        num_df = len(self.df_path_list)
        for i in range(num_df):
            df_loc = self.df_path_list[i]
            df = pd.read_pickle(df_loc)
            for index, row in df.iterrows():
                document_id = str(row['index'])  # Use a unique identifier for each document
                document_text = row['decoded_content']
                chunk_index = index % len(embedded_chunks)
                document_embedding = embedded_chunks[chunk_index]
                metadata = {'movie_name': row['name']}

                # # Insert document into ChromaDB collection
                collection.add(
                    ids=document_id, 
                    documents=[document_text], 
                    embeddings=[document_embedding], 
                    metadatas=[metadata]
                )
            logger.info(f"Insertion into ChromaDB collection complete for {i} dataframe")
        
        
