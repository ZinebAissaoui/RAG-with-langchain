# import
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import ArxivLoader
import pandas as pd 
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Q&A")
    
    data['Index'] = range(1, len(data) + 1)
    data['num_article'] = data['Lien'].str.extract(r'(\d+\.\d+)')
    return data
def process_articles(data,seperator, chroma_db_name,chunk_size,chunk_overlap,embedding_function):
    total_articles=len(data)
    
    # Créez la fonction d'embedding en dehors de la boucle
    

    error_docs = []

    for index, row in data.iterrows():
        loader = ArxivLoader(query=row['num_article'], load_max_docs=2)
        documents = loader.load()

        if documents:
            text_splitter= RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size, chunk_overlap=0)
            #text_splitter = CharacterTextSplitter(separator=seperator, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents(documents)
            for chunk in chunks:
                chunk.metadata['Summary']=str(row['Lien'])
            db2 = Chroma.from_documents(chunks, embedding_function, persist_directory=chroma_db_name)
        else:
            error_docs.append(row['num_article'])
        # Print the progress
        articles_loaded = index + 1
        if articles_loaded%5==0:
            print(f"Loaded ========={articles_loaded}/{total_articles} articles.")
    return error_docs

