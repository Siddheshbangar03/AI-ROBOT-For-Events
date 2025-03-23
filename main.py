import os
import pickle
import speech_recognition as sr
import pyttsx3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.llms import HuggingFaceHub

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# FastAPI initialization
app = FastAPI()

# PDF and vectorstore file paths
PDF_FILES = ["AIROBOT.pdf"]
VECTORSTORE_FILE = "vectorstore.pkl"

# Define request schema
class QueryRequest(BaseModel):
    query: str

# Define prompt template for LLM
TEMPLATE = """
You are an AI assistant specializing in College Event. 
Use the following context to answer the question at the end. 
If you don't know the answer, just say that you don't know. 

{context}

Question: {question}
Helpful Answer:
"""
PROMPT = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

def load_pdfs(pdf_files):
    """Load and extract text from given PDFs."""
    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyMuPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading PDF '{pdf_file}': {str(e)}")
    return documents

def create_vectorstore(documents):
    """Create FAISS vectorstore from document embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(VECTORSTORE_FILE, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def load_or_create_vectorstore():
    """Load vectorstore from disk or create a new one if missing."""
    if os.path.exists(VECTORSTORE_FILE):
        try:
            with open(VECTORSTORE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading vectorstore: {str(e)}")

    documents = load_pdfs(PDF_FILES)
    return create_vectorstore(documents)

# Load vectorstore once at startup
vectorstore = load_or_create_vectorstore()

# Load LLM from Hugging Face
llm = HuggingFaceHub(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"temperature": 0.2, "max_new_tokens": 1000, "top_p": 0.95, "repetition_penalty": 1.15}
)

def create_qa_chain(vectorstore):
    """Create QA chain for answering questions using vectorstore."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

# Initialize QA chain once
qa_chain = create_qa_chain(vectorstore)

def recognize_speech():
    """Capture voice input and convert to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand your speech."
        except sr.RequestError:
            return "Speech recognition service is not available."
        except Exception as e:
            return f"Error: {str(e)}"

def text_to_speech(text):
    """Convert text response into speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@app.get("/voice-query")
async def voice_query():
    """
    Capture voice input, process it, and return spoken response.
    """
    try:
        # Convert speech to text
        query = recognize_speech()
        
        # If speech recognition failed, return error message
        if "Sorry" in query or "Error" in query:
            return {"response": query}

        # Process query using QA model
        result = qa_chain({"query": query})
        response = result.get("result", "").strip()

        if "Helpful Answer:" in response:
            trimmed_response = response.split("Helpful Answer:")[1].strip()
        else:
            trimmed_response = response

        print(trimmed_response)

        # Speak the response
        text_to_speech(trimmed_response)
        return {"response": trimmed_response}

    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
