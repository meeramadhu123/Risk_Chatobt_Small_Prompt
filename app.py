
import streamlit as st
import pandas as pd
import os
import tempfile
import hashlib
import warnings
from PIL import Image
from datetime import datetime
import uuid
import csv

# Policy module imports
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io

# Audit module imports
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from chatbot_utils import (
    get_metadata_from_mysql,
    create_vector_db_from_metadata,
    retrieve_top_tables,
    create_llm_table_retriever,
    question_reframer,
    generate_sql_query_for_retrieved_tables,
    execute_sql_query,
    analyze_sql_query,
    finetune_conv_answer,
    debug_query,
)

warnings.filterwarnings("ignore")

OPENAI_KEY       = st.secrets["openai"]["api_key"]
DB_USER          = st.secrets["mysql"]["user"]
DB_PASSWORD      = st.secrets["mysql"]["password"]
DB_HOST          = st.secrets["mysql"]["host"]
DB_PORT          = st.secrets["mysql"]["port"]
DB_NAME          = st.secrets["mysql"]["database"]
NVIDIA_API_KEY   = st.secrets["nvidia"]["api_key"]



# -- Configurations --
logo = Image.open(r"Assets/aurex_logo.png")
descriptions_file = r"Assets/all_table_metadata.txt"
examples_file = r"Assets/Example question datasets.xlsx"

db_config = {
    "user": DB_USER,
    "password": DB_PASSWORD ,
    "host": DB_HOST,
    "port": DB_PORT,
    "database": DB_NAME
}


scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name(r"gsheets-project-458606-526d0637c909.json",scope)
client = gspread.authorize(creds)
sheet = client.open("Streamlit_Chatbot_Logs").sheet1  

headers = ["session_id","question_id","timestamp","question","sql_query",
"conversational_answer","rating", "comments"]


st.set_page_config(initial_sidebar_state='collapsed')
st.image(logo, width=150)
st.title("Welcome to Aurex AI Chatbot")
policy_flag = st.toggle("DocAI")

# Chart file hash (not used directly here)
def checkfilechange(file_path):
    with open(file_path, "rb") as f:
        newhash = hashlib.md5(f.read()).hexdigest()
    return newhash

# CSV logger
def log_csv(entry):
    log_file = "chat_log.csv"
    write_header = not os.path.exists(log_file)
    with open(log_file, "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=entry.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(entry)


def log_to_google_sheets(entry):
    """
    Appends a dictionary entry as a new row in the Google Sheet.
    """
    # Map the entry to the headers
    row = [
        entry.get("session_id", ""),
        entry.get("question_id", ""),
        entry.get("timestamp", ""),
        entry.get("question", ""),
        entry.get("sql_query", ""),
        entry.get("conversational_answer", ""),
        entry.get("rating", ""),
        entry.get("comments", "")
    ]
    
    # Append the row to the Google Sheet
    sheet.append_row(row, value_input_option="USER_ENTERED")


# Core processing, without UI
def process_risk_query(llm, user_question):
    conn, metadata = get_metadata_from_mysql(db_config, descriptions_file=descriptions_file)
    if conn is None or not metadata:
        return None, "Sorry, I was not able to connect to Database"
    vector_store = create_vector_db_from_metadata(metadata)
    docs = retrieve_top_tables(vector_store, user_question, k=10)
    top_names = [d.metadata["table_name"] for d in docs]
    example_df = pd.read_excel(examples_file)
    top3 = create_llm_table_retriever(llm, user_question, top_names, example_df)
    filtered = [d for d in docs if d.metadata["table_name"] in top3]
    reframed = question_reframer(filtered, user_question, llm)
    sql = generate_sql_query_for_retrieved_tables(filtered, reframed, example_df, llm)
    result, error = execute_sql_query(conn, sql)
    if result is None or result.empty:
        sql = debug_query(filtered, user_question, sql, llm, error)
        result, error = execute_sql_query(conn, sql)
    if result is None or result.empty:
        return "Sorry, I couldn't answer your question.",None,sql
    conv = analyze_sql_query(user_question, result.to_dict(orient='records'), llm)
    conv = finetune_conv_answer(user_question, conv, llm)
    return (conv, result, sql)

# -- Policy Module --
if policy_flag:
    st.success("Connected to Policy Module")
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if not uploaded:
        st.info("Please upload PDF documents to continue.")
        st.stop()
        
    def configure_retriever(files):
        temp = tempfile.TemporaryDirectory()
        docs = []
        for f in files:
            path = os.path.join(temp.name, f.name)
            with open(path, "wb") as out:
                out.write(f.getvalue())
            docs.extend(PyPDFLoader(path).load())
        splits = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200).split_documents(docs)
        emb = OpenAIEmbeddings(model="text-embedding-3-large", api_key= OPENAI_KEY)
        db = DocArrayInMemorySearch.from_documents(splits, emb)
        return db.as_retriever(search_type="mmr", search_kwargs={"k":2, "fetch_k":4})
    
    with st.spinner("Loading and processing documents..."):
        retriever = configure_retriever(uploaded)
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)
        llm_policy = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key= OPENAI_KEY , temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm_policy, retriever=retriever, memory=memory, verbose=False)
    
    if len(msgs.messages)==0 or st.sidebar.button("Clear history"):
        msgs.clear(); msgs.add_ai_message("How can I help you?")
        
    for m in msgs.messages:
        st.chat_message("user" if m.type=="human" else "assistant").write(m.content)
        
    if prompt := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(prompt)
        with st.spinner("Generating policy response..."):   
            handler = BaseCallbackHandler()
            resp = qa_chain.run(prompt, callbacks=[handler])
        with st.chat_message("assistant"):
            st.write(resp)

# -- Risk/Audit Module --
else:
    st.success("Connected to Risk Management Module")
    # Init LLM and session history
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'risk_msgs' not in st.session_state:
        st.session_state.risk_msgs = []
    llm_audit = ChatNVIDIA(
        model="qwen/qwen2.5-coder-32b-instruct",
        api_key= NVIDIA_API_KEY,
        temperature=0, num_ctx=50000
    )
    # Display chat history
    for msg in st.session_state.risk_msgs:
        st.chat_message(msg['role']).write(msg['content'])
    # User input at bottom
    if prompt := st.chat_input(placeholder="Ask a question about the Risk Management module"):
        # User message
        st.chat_message("user").write(prompt)
        st.session_state.risk_msgs.append({"role":"user","content":prompt})
        # Process the question
        with st.spinner("Generating the answer..."):
            conv, result, sql = process_risk_query(llm_audit, prompt)
        if conv is None:
            st.chat_message("assistant").write( "Sorry, I couldn't answer your question.")
            st.session_state.risk_msgs.append({"role":"assistant","content":"Sorry, I couldn't answer your question."})
        else:
            # Assistant response
            st.chat_message("assistant").write(conv)
            #st.dataframe(result)
            st.session_state.risk_msgs.append({"role":"assistant","content":conv})
        
            # ---- Simplified Feedback ----           
            # 1. Store the last QA in session_state so it's accessible inside the form
            st.session_state["last_prompt"] = prompt
            st.session_state["last_sql"]    = sql
            st.session_state["last_conv"]   = conv
            st.session_state["session_id"] = st.session_state.session_id
            st.session_state["question_id"] =  uuid.uuid4()
            st.session_state["timestamp"] = datetime.now().isoformat()

            # Callback to handle feedback submission
            def submit_feedback():
                entry = {
                    "session_id":   str(st.session_state["session_id"]),
                    "question_id":  str(st.session_state["question_id"]),
                    "timestamp":  str(st.session_state["timestamp"]),
                    "question": st.session_state.last_prompt,
                    "sql_query": "SQL query: "+ st.session_state.last_sql,
                    "conversational_answer": "Ans: "+ st.session_state.last_conv,
                    "rating": (1+st.session_state.feedback_rating) if st.session_state.feedback_rating else 0,
                    "comments": st.session_state.feedback_comment
                }
                if st.session_state.feedback_rating or st.session_state.feedback_comment:
                    log_to_google_sheets(entry)
                    st.success("Feedback recorded. Thank you!")	
            
                # Clear stored Q&A (optional)
                for k in ("last_prompt", "last_sql", "last_conv"):
                    st.session_state.pop(k, None)



            with st.form("feedback_form"):
                st.subheader("Rate this answer and leave optional comments")
            
                # Star rating from 1–5
                rating = st.feedback(options="stars",key="feedback_rating")
                # Text feedaback
                comment = st.text_input("Please provide comments for improvement (optional)",key="feedback_comment")
                submit = st.form_submit_button("Submit Feedback", on_click=submit_feedback)

            if submit == False:
                entry = { "session_id":   str(st.session_state["session_id"]),
                          "question_id":  str(st.session_state["question_id"]),
                          "timestamp":  str(st.session_state["timestamp"]),
                           "question":  prompt,
                           "sql_query": "SQL query: "+ sql,
                           "conversational_answer": "Ans: "+ conv,
                        }
                log_to_google_sheets(entry)
   
          
records = sheet.get_all_records()
# Convert the records to a pandas DataFrame
df = pd.DataFrame(records)
# Convert the DataFrame to CSV format in memory
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()


# Display the download button in the Streamlit sidebar
st.sidebar.markdown("### 📥 Download Chat Log")
if csv_data:
    st.sidebar.download_button(
        label="Download log (CSV)",
        data=csv_data,
        file_name="chat_log.csv",
        mime="text/csv"
    )
else:
    st.sidebar.write("No log file yet.")
