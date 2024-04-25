import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import speech_recognition as sr 
import pyttsx3
from fpdf import FPDF
from datetime import datetime

# Define your API key here
GOOGLE_API_KEY = "AIzaSyCDknvL-10pgb9hr_BP-i7JzYVIpzVHdoo"

# read all pdf files and return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )
    print(response)
    return response

def generate_pdf(messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if content:  # Check if content is not empty
            # Handle encoding errors by encoding with 'latin-1' and ignoring errors
            encoded_content = content.encode('latin-1', errors='ignore')
            pdf.multi_cell(0, 10, txt=f"{role}: {encoded_content.decode('latin-1')}")

    pdf_file_name = "chat_history.pdf"
    download_folder_path = os.path.join(os.path.expanduser("~"), "Downloads")
    pdf_file_path = os.path.join(download_folder_path, pdf_file_name)

    pdf.output(pdf_file_path)
    return pdf_file_path
    
# Function to extract basic information about PDFs
def get_pdf_info(pdf_docs):
    pdf_info = []
    for pdf in pdf_docs:
        # Get file size
        file_size = len(pdf.getvalue())
        # Get number of pages
        pdf_reader = PdfReader(pdf)
        num_pages = len(pdf_reader.pages)
        # Get number of words (approximation)
        # This can be improved for more accurate word count
        word_count = 0
        for page in pdf_reader.pages:
            word_count += len(page.extract_text().split())
        pdf_info.append({"name": pdf.name, "size": file_size, "pages": num_pages, "words": word_count})
    return pdf_info

def main():
    st.set_page_config(
        page_title="MultiPDF Chatbot",
        page_icon="📚"
    )
    # Sidebar for Documentation button
    st.sidebar.button(" View  Documentation🗒️", on_click=open_documentation)
    st.sidebar.image("img/spacer.png")
    

     # Sidebar for uploading PDF files
    with st.sidebar:
        st.write("---")
        st.image("img/Animation.gif")
        st.title("Menu:")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        # Check if files have been uploaded
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload PDF files first.")
                return  # Exit the function to prevent further processing

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        # Button to fetch basic data
        if st.button("Fetch Basic PDF Data"):
            if pdf_docs:
                basic_info = get_pdf_info(pdf_docs)
                st.write("Basic PDF Information:")
                for info in basic_info:
                    st.write(f"Name: {info['name']}, Pages: {info['pages']}, Words: {info['words']}, Size: {info['size']} bytes")
            else:
                st.warning("Please upload PDF files first.")

        clear_button_key = "clear_button_unique_key"
        st.sidebar.button('Clear Chat History', key=clear_button_key, on_click=clear_chat_history)
        
        st.write("---")
        st.sidebar.image("img/spacer.png")
        st.image("img/webapp.gif")
        st.write("""
        An AI App Developed by 
        - [👨‍💼@Anuj Jaiswal](https://www.linkedin.com/in/anuj-jaiswal-491340259)
        - [👩‍💼@Garima Tiwari](https://in.linkedin.com/in/garima-tiwari-b75182209?original_referer=https%3A%2F%2Fwww.google.com%2F)
        - [👨‍💼@Aakash Jha](https://www.linkedin.com/in/aakashjha548)
        - [👨‍💼@Rushikesh](https://in.linkedin.com/in/rushikesh-mangulkar-8b40881b5)
        """)
    
    
    # Main content area for displaying chat messages
    st.title("📚Chat with multiple PDFs📂📃💬")

    audio_path = "audio/Welcome.wav"
    if os.path.exists(audio_path):
        if st.button("Guide 📖"):
            audio_bytes = open(audio_path, 'rb').read()
            st.audio(audio_bytes, format='audio/wav', start_time=0)
        
    
    # Chat input
    # Placeholder for chat messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some documents and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input selection: speech or text
    user_input_method = st.radio("Select Interaction preference:", ("Speech", "Text"), format_func=lambda x: 'Vocal' if x == 'Speech' else ' Text')

    if user_input_method == "Speech":
        # Voice input
        if st.button("Speak Query🎙️"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Listening👂...")
                audio = recognizer.listen(source)
                try:
                    user_query = recognizer.recognize_google(audio)
                    st.session_state.messages.append(
                        {"role": "user", "content": user_query})
                    with st.chat_message("user"):
                        st.write(user_query)
                    with st.spinner("Thinking🤔💭..."):
                        response = user_input(user_query)
                        placeholder = st.empty()
                        full_response = ''
                        for item in response['output_text']:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)
                    if response is not None:
                        message = {"role": "assistant",
                                   "content": full_response}
                        st.session_state.messages.append(message)
                except sr.UnknownValueError:
                    st.error("Sorry, I could not understand what you said.")
                except sr.RequestError as e:
                    st.error(
                        f"Could not request results from Google Speech Recognition service; {e}")

        if st.button("Auditory Feedback🔉"):
            last_response = st.session_state.messages[-1]["content"]
            text_to_speech(last_response)

    elif user_input_method == "Text":
        # Text input
        if prompt := st.chat_input("Enter your query✍️:"):
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            with st.spinner("Thinking🤔💭.."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

    # Black strip with GitHub link
    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #A9A9A9; padding: 15px; text-align: center; color: #282434;">
            Made with the help of LangChain🦜 & Gemini💡 <a href="https://github.com/Anujj-Jaiswal/Chat-with-Multiple-PDFs" target="_blank">Github</a>🔗 
        </div>

        """,
        unsafe_allow_html=True
    )
    if st.button("Download Chat 💾"):
        pdf_file_path = generate_pdf(st.session_state.messages)
        st.success(f"Chat history downloaded as {pdf_file_path}")
    
def open_documentation():
    import webbrowser
    url = "https://anujj-jaiswal.github.io/MultiPDF-Chatbot-Documentation/"  # Replace with your documentation URL
    webbrowser.open_new_tab(url)

# Function to convert text to speech using gTTS and play it in the Streamlit app
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    # Create a BytesIO object to store audio data in memory
    audio_bytes = BytesIO()
    # Save the audio to the BytesIO object
    tts.write_to_fp(audio_bytes)
    # Reset the BytesIO object to the beginning
    audio_bytes.seek(0)
    # Play the audio in the Streamlit app
    st.audio(audio_bytes, format='audio/mp3', start_time=0)

if __name__ == "__main__":
    main()
