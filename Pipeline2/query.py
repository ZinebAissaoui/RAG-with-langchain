import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

def query_chroma_with_gemini(question,embedding_function,chroma_path):
    # Sentence embeddings configuration

    # 1. Configuration
    genai.configure(api_key="***************")
    generation_config = {"temperature": 0.5, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}

    # 2. Initialize the model
    model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)

    # Prepare the DB
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Sample query text
    query_text = question

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    # Extract context text from results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Create prompt for the generative model
    prompt = f""" Answer the question based only on the following context: {context_text}
    ---

    Answer the question based on the above context: {query_text}"""
    
    # Generate responses using the model
    responses = model.generate_content(prompt)
    
    # Access the content using result.parts or result.candidates[index].content.parts
    parts = responses.parts if hasattr(responses, 'parts') else responses.candidates[0].content.parts

    # Print the generated responses
    return(parts[0].text)


