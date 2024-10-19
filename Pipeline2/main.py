from query import query_chroma_with_gemini
from create_langchain_database_chroma import process_articles, load_data
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

import pandas as pd
#Choosing our embedding model
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chunk_size=1000 #size of chunks
chunk_overlap=0 #size of overlaps
seperator='\n'  #separator of chunks
chroma_db_name='C:/Users/zaiss/OneDrive - Ecole Centrale Casablanca/Hackathon/chroma_articles' # directory to store database 
# function which read articles, chunks, embedding and store on a chroma database
file_path = "C:/Users/zaiss/OneDrive - Ecole Centrale Casablanca/Hackathon/Pipeline1/Lien_articles_Z.xlsx"
data=load_data(file_path)
#errors = process_articles(data,seperator, chroma_db_name,chunk_size,chunk_overlap,embedding_function)
# Maintenant, errors contient la liste des numéros d'articles qui n'ont pas pu être traités.
#print(f"An error occured while reading those articles: {errors}")

#### querying###
for i, row in data.iterrows():
    question=str(data["QUESTION"][i])
    
    response=query_chroma_with_gemini(question, embedding_function, chroma_db_name)
    data.loc[i,'RéponsesPipeline2']=str(response)

# function which read articles, chunks, embedding and store on a chroma database
df_liens=load_data(file_path)
df_merged = pd.merge(df_liens, data[['Lien', 'RéponsesPipeline2']], on='Lien', how='left')
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
    # Écrire le dataframe fusionné dans une nouvelle feuille
    df_merged.to_excel(writer, sheet_name='Pipeline2', index=False) 
