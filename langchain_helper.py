import os 
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


load_dotenv()



llm = GooglePalm(google_api_key=os.environ["API_KEY"], temperature=0)

instruct_embeddings = HuggingFaceInstructEmbeddings()

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path= "Chatbot_Prompts - Sheet1.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instruct_embeddings)
    vectordb.save_local("faiss_index")
    
    
def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instruct_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    
    prompt_template = """Given the following context and a question, generate an answer based on the context of given information csv. Use direct information from the provided context where possible and avoid extrapolation or guesses.

    Note: This chatbot is designed to share insights about Kunal Chopra's professional background and his REU project. It may not have updates on all projects or comprehensive knowledge beyond these areas.

    CONTEXT: {context}

    QUESTION: {question}

    If the information sought is within the chatbot's knowledge domain, it will provide an answer based on the 'response' section in the context. If not, or if the answer is not explicitly covered in the context, the chatbot will respond with "I don't know." Please refrain from making up an answer outside the given context.
    """


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    
    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True)
    return chain

    

    
if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain("What is Kunal's major?"))
    

