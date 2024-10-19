
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

def query_chroma_with_openai(question, embedding_function, chroma_path="./chroma_articles", api_key=""):
    os.environ["OPENAI_API_KEY"] = api_key

    CHROMA_PATH = chroma_path
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}
    ---

    Answer the question based on the above context: {question}
    """

    # Prepare the DB.
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(question, k=10)
  
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    sources = [doc.metadata.get("Summary", None) for doc, _score in results]
    unique_sources = list(set(sources))
    if len(unique_sources)>=2:
        source=unique_sources[:2]
    elif len(unique_sources)==1:
        source=unique_sources[0]
    else:
        source='None'
    

    return response_text,source


