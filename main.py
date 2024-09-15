
#pip install streamlit python-dotenv langchain faiss-cpu transformers sentence-transformers python-docx unstructured[local-inference]

import os 
import tempfile
import subprocess
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from docx import Document

# Functions to extract text from different document types
def extract_text_from_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path)
    loaded_data = loader.load()
    return ''.join([doc.page_content for doc in loaded_data])

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def get_text_from_docs(docs):
    data = ""
    for doc in docs:
        # Get the file name and extension
        file_name = doc.name.lower()
        
        # Create a temporary file for the uploaded document
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(doc.read())
            temp_file_path = temp_file.name
        
        # Determine the file type and extract text accordingly
        if file_name.endswith('.pdf'):
            data += extract_text_from_pdf(temp_file_path)
        elif file_name.endswith('.docx'):
            data += extract_text_from_docx(temp_file_path)
        elif file_name.endswith('.txt'):
            data += extract_text_from_txt(temp_file_path)
        else:
            os.remove(temp_file_path)  # Clean up the temporary file
            raise ValueError(f"Unsupported file format: {file_name}")
        
        # Clean up the temporary file
        os.remove(temp_file_path)

    return data

# Function to split text into chunks for processing
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vectorstore(text_chunks, model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1"):
    # Use a more powerful embedding model
    try:
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        # Suppress the warning by setting protected_namespaces
        embedding_model.model_config['protected_namespaces'] = ()
    except AttributeError:
        # In case the model does not have the model_config attribute
        pass

    embeddings = embedding_model.embed_documents(text_chunks)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    return vectorstore

# Function to get the conversation chain 
def get_conversation_chain(vectorstore, use_local_model=False):
    if use_local_model:
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hf_pipeline = pipeline(
            "text2text-generation", 
            model=model, 
            tokenizer=tokenizer,
            # Adjust these parameters for more verbose responses
            max_length=1024,       # Allow longer responses
            temperature=0.7,       # Increase temperature for more creative responses
            top_p=0.9,             # Nucleus sampling for more diverse outputs
            num_beams=3            # Beam search for more coherent output
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)
    else:
        # Using HuggingFaceHub with adjusted parameters for verbose responses
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large", 
            model_kwargs={
                "temperature": 0.7,  # Adjust temperature for creativity
                "max_length": 1024,  # Allow longer responses
                "top_p": 0.9,        # Use nucleus sampling
                "num_beams": 3       # Beam search for better quality
            }
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

# Function to handle user input and generate a more verbal response
def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation.invoke({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            message_content = str(message.content) if hasattr(message, 'content') else str(message)
            if i % 2 == 0:
                # User message
                st.write(user_template.replace("{{MSG}}", message_content), unsafe_allow_html=True) 
            else:
                formatted_response = f"Let me see... {message_content}"
                st.write(bot_template.replace("{{MSG}}", formatted_response), unsafe_allow_html=True)
    else:
        st.warning("Please process the documents first before asking questions.")

# Main Streamlit app logic
def main():
    load_dotenv()
    
    # Automatically run the Streamlit app on port 8505 (just in case...)
    command = "streamlit run main.py --server.port 8505"
    subprocess.run(command, shell=True)

    st.set_page_config(page_title="Ask the wizard about your files")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state: 
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask the wizard about your files.")
    user_question = st.text_input("Load, process, and locally ask questions about your text documents.")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader(
            "Upload your documents here and click on 'Process' (keep in mind heavier files mean longer waiting times in weaker setups)", 
            accept_multiple_files=True, 
            type=["pdf", "docx", "txt"]
        )
        
        if st.button("Process"):
            if uploaded_files:
                with st.spinner("Processing..."):
                    raw_text = get_text_from_docs(uploaded_files)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks, model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

                    # Set up the conversation chain (using a smaller model)
                    st.session_state.conversation = get_conversation_chain(vectorstore, use_local_model=False)
            else:
                st.warning("Please upload files first.")

if __name__ == '__main__':
    main()





