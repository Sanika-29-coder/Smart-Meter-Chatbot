"""
Smart Home Energy Advisor - Streamlit RAG Chatbot using Groq
Features:
 - Upload PDFs (bills/reports/manuals) -> RAG retrieval over document content
 - Upload CSV meter data -> quick usage analysis (peaks, avg, recommended run-times)
 - Combine retrieved context + meter analysis to answer user questions using Groq LLM (ChatGroq)
 - Uses GROQ_API_KEY from environment (you said it's already in env)
"""
from dotenv import load_dotenv
load_dotenv()

import os
import io
import warnings
import logging
from datetime import timedelta

import streamlit as st
import pandas as pd
import numpy as np

# LangChain / embeddings / loaders
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


# Silence noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Smart Home Energy Advisor", layout="wide")
st.title("🔌 Smart Home Energy Advisor — RAG Chatbot")

# === Sidebar: Upload files and settings ===
st.sidebar.header("Upload data & settings")
pdf_files = st.sidebar.file_uploader(
    "Upload PDF(s) (bills, manuals) — used to build knowledge base",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload monthly bills, installation manuals, or energy reports."
)

csv_file = st.sidebar.file_uploader(
    "Upload meter CSV (timestamp, usage_kWh) — optional", type=["csv"], help="CSV with timestamp & usage columns."
)

model_name = st.sidebar.text_input("Groq model name", value=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.sidebar.error("GROQ_API_KEY not found in environment. Set GROQ_API_KEY in your env vars.")

k_retriever = st.sidebar.slider("Retriever: top k docs", min_value=1, max_value=6, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick tips**")
st.sidebar.write(
    "- Upload recent bills or device manuals (PDF) for best answers.\n"
    "- Upload meter CSV if you want time-of-day analysis and specific recommendations."
)

# === Session state for messages and vectorstore ===
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'meter_analysis' not in st.session_state:
    st.session_state.meter_analysis = None
if 'docs_ingested' not in st.session_state:
    st.session_state.docs_ingested = False

# === Helpers ===
@st.cache_resource
def build_pdf_vectorstore(uploaded_pdf_files):
    """Create a vectorstore index from uploaded PDFs and return vectorstore."""
    if not uploaded_pdf_files:
        return None

    try:
        local_paths = []
        for uploaded in uploaded_pdf_files:
            local_path = f"./tmp_{uploaded.name}"
            with open(local_path, "wb") as f:
                f.write(uploaded.read())
            local_paths.append(local_path)

        # Load documents
        loaders = [PyPDFLoader(p) for p in local_paths]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120
        )
        docs = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

        # Build vectorstore
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_documents(docs, embeddings)

        return vectorstore

    except Exception as e:
        raise ValueError(f"Failed to process PDFs: {e}")


    # Try to detect timestamp column
    timestamp_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    usage_cols = [c for c in df.columns if "usage" in c.lower() or "kwh" in c.lower() or "consumption" in c.lower() or "value" in c.lower()]

    # fallback heuristics
    if not timestamp_cols:
        # try first column if it looks like datelike
        timestamp_cols = [df.columns[0]]
    if not usage_cols:
        # try last column
        usage_cols = [df.columns[-1]]

    ts_col = timestamp_cols[0]
    usage_col = usage_cols[0]

    # Parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col])
    df[usage_col] = pd.to_numeric(df[usage_col], errors='coerce').fillna(0.0)

    # If timestamps are not sorted, sort
    df = df.sort_values(ts_col)

    # Resample to hourly consumption if timestamps contain higher resolution
    df = df.set_index(ts_col)
    # If intervals are already hourly or coarser, this will produce reasonable aggregation
    hourly = df[usage_col].resample('H').sum().fillna(0.0)

    total_kwh = hourly.sum()
    avg_hour = hourly.mean()
    peak_kwh = hourly.max()
    peak_hour = int(hourly.idxmax().hour) if not hourly.empty else None

    # Daily aggregates
    daily = hourly.resample('D').sum()
    avg_daily = daily.mean() if not daily.empty else None

    # Hour-of-day average profile: to suggest best run times
    profile = hourly.groupby(hourly.index.hour).mean()  # mean across same hour of day

    # Suggest off-peak hours as the hours with lowest average in the profile
    sorted_profile = profile.sort_values()
    offpeak_hours = sorted_profile.index.tolist()[:3]  # three best hours
    peak_hours = sorted_profile.index.tolist()[-3:][::-1]  # three worst hours (highest)

    # Build human-friendly summary
    summary = {
        "total_kwh": float(total_kwh),
        "avg_hourly_kwh": float(avg_hour) if avg_hour is not None else None,
        "peak_kwh": float(peak_kwh) if peak_kwh is not None else None,
        "peak_hour": peak_hour,
        "avg_daily_kwh": float(avg_daily) if avg_daily is not None else None,
        "offpeak_hours": [int(h) for h in offpeak_hours],
        "peak_hours": [int(h) for h in peak_hours],
        "profile_series": profile,  # pandas Series: avg usage by hour-of-day
        "hourly_series": hourly
    }
    return summary

def format_meter_summary_for_prompt(summary):
    """Turn the meter analysis dict into a brief textual summary to include in system/context."""
    if not summary:
        return ""
    lines = []
    lines.append(f"Meter data summary:")
    lines.append(f"- Total energy in provided period: {summary['total_kwh']:.2f} kWh.")
    if summary.get('avg_daily_kwh') is not None:
        lines.append(f"- Average daily consumption: {summary['avg_daily_kwh']:.2f} kWh.")
    if summary.get('peak_kwh') is not None and summary.get('peak_hour') is not None:
        lines.append(f"- Highest hourly consumption: {summary['peak_kwh']:.2f} kWh at hour {summary['peak_hour']}:00.")
    lines.append(f"- Typical off-peak hours (based on your data): {', '.join(str(h)+':00' for h in summary['offpeak_hours'])}.")
    lines.append(f"- Typical peak hours (based on your data): {', '.join(str(h)+':00' for h in summary['peak_hours'])}.")
    lines.append("Use this data when suggesting appliance run-times and energy-saving recommendations.")
    return "\n".join(lines)

# === Ingest uploaded PDFs to vectorstore (if any) ===
if pdf_files:
    with st.spinner("Ingesting PDFs and building vectorstore..."):
        try:
            st.session_state.vectorstore = build_pdf_vectorstore(pdf_files)
            st.session_state.docs_ingested = True
            st.success(f"Ingested {len(pdf_files)} PDF(s) and created vectorstore.")
        except Exception as e:
            st.session_state.vectorstore = None
            st.session_state.docs_ingested = False
            st.error(f"Failed to build vectorstore: {e}")

# === Parse CSV meter (if any) and show quick charts ===
if csv_file:
    try:
        csv_bytes = csv_file.read()
        csv_buffer = io.BytesIO(csv_bytes)
        meter_summary = analyze_meter_csv(csv_buffer)
        st.session_state.meter_analysis = meter_summary

        # Plot hourly series (interactive)
        st.subheader("Meter data — hourly consumption")
        # Convert hourly_series to DataFrame for streamlit chart
        hr_df = meter_summary['hourly_series'].rename("kWh").reset_index().set_index(meter_summary['hourly_series'].index)
        st.line_chart(meter_summary['hourly_series'])

        st.markdown("**Quick meter summary**")
        st.write(format_meter_summary_for_prompt(meter_summary))
    except Exception as e:
        st.error(f"Failed to parse meter CSV: {e}")

# === Chat UI ===
st.markdown("---")
st.subheader("Ask your Smart Home Energy Advisor")
user_input = st.chat_input("Ask about your bill, appliance scheduling, or how to save energy...")

# Display historical messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).markdown(msg['content'])

if user_input:
    # show user's message in chat
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    # Prepare Groq LLM
    if not groq_api_key:
        st.error("GROQ_API_KEY missing - can't call model.")
    else:
        try:
            groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

            # Compose a system prompt specialized for Smart Home Energy Advisor
            # Include meter summary if available
            meter_text = ""
            if st.session_state.meter_analysis:
                meter_text = format_meter_summary_for_prompt(st.session_state.meter_analysis)

            system_prompt_template = ChatPromptTemplate.from_template(
                """You are a helpful Smart Home Energy Advisor. Your job is to:
1) Use any retrieved document passages (from user's uploaded PDFs) and any meter data summary (if provided) to produce accurate, actionable, and concise recommendations to save electricity and lower bills.
2) Provide simple explanations, steps the user can follow, and approximate expected savings if reasonable.
3) When asked about scheduling (best time to run appliances), prefer hours marked as off-peak in the meter summary or ask about time-of-use (TOU) rates if missing.
4) If the question requires external info (like exact TOU rates), say you don't have live tariff data and recommend the user provide tariff rates or a bill PDF.
5) If the user asks for step-by-step analysis, show calculations and mention assumptions.

Context (meter summary, if provided):
{meter_summary}

Retrieved documents (if any) should be used to support answers. Start the answer directly, be concise, and avoid filler.
User question: {user_prompt}
"""
            )

            # If vectorstore exists, use RetrievalQA chain to combine retrieval + LLM
            if st.session_state.vectorstore is not None:
                # Build retrieval QA chain
                chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type='stuff',
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': k_retriever}),
                    return_source_documents=True
                )

                # Prepare the final prompt by substituting variables
                prompt_inputs = {
                    "meter_summary": meter_text,
                    "user_prompt": user_input
                }
                # Run chain (it will use groq_chat as LLM)
                with st.spinner("Thinking with RAG + Groq..."):
                    result = chain({"query": system_prompt_template.format_prompt(**prompt_inputs).to_messages()[-1].content + "\n\n" + user_input})

                # `result` should contain 'result' and 'source_documents' depending on chain implementation
                if isinstance(result, dict) and "result" in result:
                    assistant_response = result["result"]
                else:
                    assistant_response = str(result)

                # Show assistant response and sources
                st.chat_message("assistant").markdown(assistant_response)
                st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})

                # Show top sources (if any)
                if isinstance(result, dict) and result.get("source_documents"):
                    #st.markdown("**Sources used:**")
                    for doc in result["source_documents"][:k_retriever]:
                        meta = getattr(doc, "metadata", {})
                        srcname = meta.get("source") or meta.get("filename") or "Document"
                        snippet = doc.page_content[:400].replace("\n", " ")
                        st.write(f"- **{srcname}** — {snippet}...")
            else:
                # No docs: call LLM directly but include meter summary in prompt
                # Build prompt text
                prompt_text = system_prompt_template.format_prompt(meter_summary=meter_text, user_prompt=user_input).to_messages()[-1].content
                with st.spinner("Generating answer from Groq..."):
                    assistant_response = groq_chat.generate([{"role": "user", "content": prompt_text}]).generations[0][0].text

                st.chat_message("assistant").markdown(assistant_response)
                st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})

        except Exception as e:
            st.error(f"Error calling Groq/chain: {e}")
