# Resume
The goal of this project is to build a RAG (Retrieval-Augmented Generation) pipeline that allows us to extract precise answers from selected Arxiv articles. 
This solution can be adapted for other sources, whether PDF or non-PDF, while maintaining the same project steps.

We tested multiple pipelines by combining various types of parsers (such as Langchain’s ArxivLoader and LlmSherpa), different databases (Chroma, JSON, and LlamaIndex) to search within chunks, and various embedders (like Roberta, SentenceTransformer, and Text-Embedding Ada-002, among others). 
Finally, we evaluated several LLMs (GPT 3.5, 4, 4 Turbo, and Gemini). The results and the details of the combinations are presented in the attached PDF presentation.

# The RAG processus


**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with language generation models to provide more accurate and contextually relevant answers. Instead of relying solely on pre-trained language models, RAG enhances their performance by first retrieving relevant documents or data from an external knowledge base (such as PDFs, databases, or websites) and then using that retrieved information to generate responses.

The process typically involves two steps:

1. **Retrieval**: The model retrieves relevant chunks of information from a predefined dataset or knowledge source based on a query or prompt.
  
2. **Generation**: Using the retrieved information, a language model (like GPT or any other LLM) generates a coherent and precise response.

RAG improves the quality of answers, especially in scenarios where the language model alone might lack specific or up-to-date knowledge. This method is particularly useful for tasks such as answering questions from long documents, technical papers, or industry-specific data.

![image](https://github.com/user-attachments/assets/8b2c3fbd-0d78-4849-a647-cc85e80cd540)

# Benchmark

![image](https://github.com/user-attachments/assets/9be034a9-b7d3-4bf5-962d-b1fc89f6ef16)

# Pipelines 

![image](https://github.com/user-attachments/assets/bc40e45e-30d1-4974-91af-d5ce1d294506)

In this project, you will find the script for **pipelines 1 and 2**, which represent my contributions to the hackathon.

# Results
1. 
![image](https://github.com/user-attachments/assets/fbd8624f-a32f-431c-9e3d-8b57735de3e3)

2.
![image](https://github.com/user-attachments/assets/7343521b-ec06-4499-853a-0ee274f1c44f)

## Proof of concept
![image](https://github.com/user-attachments/assets/2f137408-977f-414f-98d8-35b8e9ffe792)











