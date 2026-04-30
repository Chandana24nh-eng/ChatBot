import streamlit as st
import pandas as pd
import os
import re
import sqlite3
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI SQL Analyst Agent")
st.title("🗄️ AI SQL Analyst Agent")

# ---------------- LOAD API ---------------- #
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# ---------------- CLEAN CODE ---------------- #
def clean_code(code):
    code = re.sub(r"```sql", "", code)
    code = code.replace("```", "")
    return code.strip()

# ---------------- FILE UPLOAD ---------------- #
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # 🔥 CREATE SQLITE DB
    conn = sqlite3.connect("temp.db")
    df.to_sql("data", conn, if_exists="replace", index=False)

    question = st.text_input("Ask a question about your data")

    if question:

        # 🔥 SINGLE PROMPT (ONLY SQL)
        prompt = f"""
You are a senior data analyst.

Table name: data
Columns: {list(df.columns)}

Write ONLY SQL query to answer the question.

Rules:
- Use table name 'data'
- No explanation
- No markdown

Question: {question}
"""

        response = llm.invoke(prompt)

        sql_query = clean_code(response.content)

        st.subheader("🧠 Generated SQL")
        st.code(sql_query)

        # 🔥 EXECUTION
        try:
            result = pd.read_sql_query(sql_query, conn)

            st.subheader("📊 Answer")
            st.write(result)

        except Exception as e:
            st.error(f"Execution Error: {e}")