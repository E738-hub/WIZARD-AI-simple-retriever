================================================================================================================
OVERVIEW
================================================================================================================
This project is a lightweight, local AI that allows users to upload documents and ask simple, brief questions based on the content of those documents. The AI runs entirely on your computer; no information leaves your PC, ensuring privacy and security of your data. Given that this is a prototype designed to run on laptops and devices with limited computational resources, it will not perform at the same level as cloud-based AI models like ChatGPT or other large-scale AI systems. It's best suited for straightforward queries rather than complex or deep analysis.


================================================================================================================
HOW IT WORKS
================================================================================================================
Upload Documents: You can upload PDF, DOCX, and TXT files. The AI will extract text from these documents to create a searchable knowledge base.

Text Processing: The extracted text is split into smaller chunks to optimize the search and retrieval process.

Vector Store Creation: The text chunks are converted into embeddings (vector representations) using a sentence transformer model. These vectors are then stored locally using FAISS, a lightweight and efficient vector database.

Question Answering: When you ask a question, the AI retrieves the most relevant chunks of text from the vector store and uses a language model (either local or a lightweight version from Hugging Face) to generate a response based on the content.

Local Processing: The entire process, including document analysis and question answering, runs locally on your computer. No data is sent to external servers, ensuring that your documents and questions remain private.


================================================================================================================
LIMITATIONS
================================================================================================================
This AI is designed for lightweight setups, such as laptops, and will not perform at the level of more powerful AIs like ChatGPT.
It's best suited for simple, brief questions rather than extensive or detailed inquiries.
Processing time may vary depending on the size of the documents and the computational power of your device.
Dependencies
To run this project, you need to install the following Python packages:
streamlit python-dotenv langchain faiss-cpu transformers sentence-transformers python-docx

just run this command on your console:

"pip install streamlit python-dotenv langchain faiss-cpu transformers sentence-transformers python-docx"




================================================================================================================
USAGE INSTRUCTIONS
================================================================================================================
Run the Application:

Execute the main script (main.py). The Streamlit app will automatically open in your web browser on port 8505.
Upload Documents:

Use the sidebar to upload your documents. Supported file formats are PDF, DOCX, and TXT.
Click the "Process" button to extract text and build the knowledge base from the uploaded files.
Ask Questions:

After processing, type your questions in the text input field at the top of the application.
The AI will provide responses based on the content of the uploaded documents.
Privacy Assurance:

All processing happens locally on your computer. No information is sent to the cloud or external servers.
Notes
If you wish to use a local language model for question-answering, the script supports a model from Hugging Face (google/flan-t5-large).
For improved privacy, ensure you only run this application on a trusted device.
This project is meant for experimentation and prototyping, so performance and response quality may vary depending on your hardware capabilities.
