import streamlit as st
from streamlit.components.v1 import html
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
import json


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
descriptions_file = r"all_table_metadata_v2.txt"
examples_file = r"Assets/Example question datasets.xlsx"

db_config = {
    "user": DB_USER,
    "password": DB_PASSWORD ,
    "host": DB_HOST,
    "port": DB_PORT,
    "database": DB_NAME
}


scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
# Convert st.secrets to a JSON-style dict
creds_dict = dict(st.secrets["gsheets"])
# Convert to actual JSON string and parse it
creds_json = json.loads(json.dumps(creds_dict))
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json,scope)
client = gspread.authorize(creds)
sheet = client.open("Streamlit_Chatbot_Logs").sheet1  

headers = ["session_id","question_id","timestamp","question","sql_query",
"conversational_answer","rating", "comments"]


st.set_page_config(initial_sidebar_state='expanded')
st.image(logo, width=150)
st.title("Welcome to Aurex AI Chatbot")
policy_flag = st.toggle("DocAI")

# 2. Sidebar expander for intermediate steps
with st.sidebar:
    st.markdown("### ⚙️ Intermediate Steps")
    steps_expander = st.expander("Show steps", expanded=False)
    step_titles = [
        "Top 10 Tables",
        "Top 3 Tables via LLM",
        "Reframed Question",
        "Generated SQL",
        "Debugged SQL",
        "Query Result Sample",
        "Initial Conversational Draft"
    ]
    placeholders = {title: steps_expander.container() for title in step_titles}


class PrintRetrievalHandler(BaseCallbackHandler):
        def __init__(self, container):
            self.status = container.status("**Context Retrieval**")

        def on_retriever_start(self, serialized: dict, query: str, **kwargs):
            self.status.write(f"**Question:** {query}")
            self.status.update(label=f"**Context Retrieval:** {query}")

        def on_retriever_end(self, documents, **kwargs):
            for idx, doc in enumerate(documents):
                source = os.path.basename(doc.metadata["source"])
                self.status.write(f"**Document {idx} from {source}**")
                self.status.markdown(doc.page_content)
            self.status.update(state="complete")



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
    # Check if 'conn' and 'vector_store' are already in session state
    if 'conn' not in st.session_state or 'vector_store' not in st.session_state:
        with st.spinner("🔍 Connecting to the Risk management database..."):
            # Establish the database connection and create the vector store
            conn, metadata = get_metadata_from_mysql(db_config, descriptions_file=descriptions_file)
        with st.spinner("🔍 Connecting to the vector database..."):
            vector_store = create_vector_db_from_metadata(metadata)
            # Store them in session state
            st.session_state.conn = conn
            st.session_state.metadata = metadata
            st.session_state.vector_store = vector_store
    else:
        # Retrieve from session state
        conn = st.session_state.conn
        metadata = st.session_state.metadata
        vector_store = st.session_state.vector_store
        
    if conn is None or not metadata:
            return "Sorry, I was not able to connect to Database", None, ""
    with st.spinner("📊 Retrieving the metadata for most relevant tables..."):
        docs = retrieve_top_tables(vector_store, user_question, k=10)
        top_names = [d.metadata["table_name"] for d in docs]
        placeholders["Top 10 Tables"].markdown("## Top 10 Tables after stage 1 retrieval")
        placeholders["Top 10 Tables"].write(", ".join(top_names))
        example_df = pd.read_excel(examples_file)
        top3 = create_llm_table_retriever(llm, user_question, top_names, example_df)
        placeholders["Top 3 Tables via LLM"].markdown("## Top 3 Tables after stage 2 retrieval")
        placeholders["Top 3 Tables via LLM"].write(top3)
        filtered = [d for d in docs if d.metadata["table_name"] in top3]

    with st.spinner("📝 Reframing question based on metadata..."):
        reframed = question_reframer(filtered, user_question, llm)
        placeholders["Reframed Question"].markdown("## Question Rephrasing Process")
        placeholders["Reframed Question"].write(reframed)

    with st.spinner("🛠️ Generating SQL query..."):
        sql = generate_sql_query_for_retrieved_tables(filtered, reframed, example_df, llm)
        placeholders["Generated SQL"].markdown("## SQL Query Generation Process")
        placeholders["Generated SQL"].code(sql)
        
    with st.spinner("🚀 Executing SQL query..."):
        result, error = execute_sql_query(conn, sql)
        if result is None or result.empty:
            with st.spinner("🧪 Debugging SQL query..."):
                sql = debug_query(filtered, user_question, sql, llm, error)
                result, error = execute_sql_query(conn, sql)
                placeholders["Debugged SQL"].markdown("## SQL Query Debugging Process")
                placeholders["Debugged SQL"].code(sql)
            if result is None or result.empty:
                return "Sorry, I couldn't answer your question.", None, sql
        placeholders["Query Result Sample"].markdown("## Tabular Result of SQL Query")        
        #placeholders["Query Result Sample"].table(result)
        placeholders["Query Result Sample"].dataframe(result, width=600, height=300)
        try:
            placeholders["Query Result Sample"].dataframe(result, width=600, height=300)
        except ValueError as e:
            # detect and drop duplicate columns
            if "Duplicate column names found" in str(e):
                result = result.loc[:, ~result.columns.duplicated()]
                placeholders["Query Result Sample"].dataframe(result, width=600, height=300)
            else:
                return "Sorry, I couldn't answer your question.", None, sql

    with st.spinner("📈 Analyzing SQL query results..."):
        conv = analyze_sql_query(user_question, result.to_dict(orient='records'), llm)
        placeholders["Initial Conversational Draft"].markdown("## Initial Answer before finetuning process")
        placeholders["Initial Conversational Draft"].write(conv)

    with st.spinner("💬 Finetuning conversational answer..."):
        conv = finetune_conv_answer(user_question, conv, llm)

    return conv, result, sql


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
            retrieval_handler = PrintRetrievalHandler(st.container())
            resp = qa_chain.run(prompt, callbacks=[handler, retrieval_handler])
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
    #llm_audit = ChatNVIDIA(model="qwen/qwen2.5-coder-32b-instruct",api_key= NVIDIA_API_KEY,temperature=0, num_ctx=50000)
    llm_audit = ChatNVIDIA(model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",base_url="http://54.161.46.7/v1/",temperature=0,max_tokens=1024, top_p=0.1,seed=42)
    
    # Display chat history
    for msg in st.session_state.risk_msgs:
        st.chat_message(msg['role']).write(msg['content'])
    # User input at bottom
    if prompt := st.chat_input(placeholder="Ask a question about the Risk Management module"):
        # User message
        st.chat_message("user").write(prompt)
        st.session_state.risk_msgs.append({"role":"user","content":prompt})
        # Process the question
        #with st.spinner("Generating the answer..."):
        conv, result, sql = process_risk_query(llm_audit, prompt)
        if conv is None:
            st.chat_message("assistant").write( "Sorry, I couldn't answer your question.")
            st.session_state.risk_msgs.append({"role":"assistant","content":"Sorry, I couldn't answer your question."})
        else:
            # Assistant response
            #st.chat_message("assistant").write(conv)
            tab1, tab2 = st.tabs(["Conversational", "Tabular"])
            tab1.chat_message("assistant").write(conv)
            tab2.dataframe(result,width=600, height=300)
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

            
            feedback_expander = st.expander("Give Feedback", expanded=False)
            with feedback_expander:
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
