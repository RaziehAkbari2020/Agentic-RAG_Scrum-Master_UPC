
# -*- coding: utf-8 -*-
"""Agentic RAG — Streamlit App (Router → Retrieve → Grade → Rewrite/Generate)"""

import os, json, uuid
from pathlib import Path
import streamlit as st

# ------------------ Basic page config ------------------
if not st.session_state.get("_pg_cfg_set"):
    st.set_page_config(page_title="Agentic RAG — Intelligent Assistant for Agile Project Management,As an Agile Project Manager — Chat with your data", page_icon="💬")
    st.session_state["_pg_cfg_set"] = True

st.image("upc.png", width=150)
st.markdown("""
<h2 style='text-align: center; color: #1c1c1c;'>
<b>Agentic RAG — Intelligent Assistant for Agile Project Management</b>
</h2>
<h4 style='text-align: center; color: #555555; font-weight: normal;'>
As an Agile Project Manager — Chat with your data
</h4>
""", unsafe_allow_html=True)

# ------------------ LangChain / LangGraph imports ------------------
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from pydantic import BaseModel, Field
from typing import Literal

# ------------------ API key ------------------
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not (OPENAI_API_KEY and (OPENAI_API_KEY.startswith("sk-") or OPENAI_API_KEY.startswith("sk-proj-"))):
    st.error("OPENAI_API_KEY در Secrets/Environment تنظیم نشده یا معتبر نیست.")
    st.stop()

# ------------------ Upload JSONL ------------------
up = st.file_uploader("Upload your data (JSONL: each line with keys {page_content, metadata})", type=["jsonl"])
if not up:
    st.info("Please upload your documents.jsonl file to continue.")
    st.stop()

# Reset vector index if a different file arrives
upload_sig = (up.name, getattr(up, "size", None))
if st.session_state.get("last_upload_sig") != upload_sig:
    st.session_state.pop("vector_store", None)
    st.session_state["last_upload_sig"] = upload_sig

# Read documents
documents = []
for line in up:
    obj = json.loads(line.decode("utf-8"))
    documents.append(Document(page_content=obj["page_content"], metadata=obj.get("metadata", {})))

# ------------------ Chunking ------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\\n\\n", "\\n", ".", " "],
)
chunks = text_splitter.split_documents(documents)
st.caption(f"🔹 Total chunks created: {len(chunks)}")

# ------------------ Vector store (FAISS) ------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = FAISS.from_documents(chunks, embeddings)
vector_store = st.session_state["vector_store"]

# ------------------ Retriever tool ------------------
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve user stories and related chunks relevant to Agile PM queries."""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\\n\\n".join(
        (
            f"🔹 US-ID: {doc.metadata.get('us_id')}\\n"
            f"📄 Content:\\n{doc.page_content}"
        )
        for doc in retrieved_docs
    )
    # content for LLM, artifact with the raw docs for later grading/generation
    return serialized, retrieved_docs

tools_node = ToolNode([retrieve])

# ------------------ LLMs ------------------
llm_router = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=OPENAI_API_KEY, temperature=0)
llm_grader = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=OPENAI_API_KEY, temperature=0)
llm_answer = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=OPENAI_API_KEY, temperature=0)

# ------------------ Router (decide to call tool or not) ------------------
def query_or_respond(state: MessagesState):
    """Bind the retriever tool and let the model decide whether to call it."""
    llm_with_tools = llm_router.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ------------------ Grader (binary relevance) ------------------
GRADE_PROMPT = (
    "You are a grader assessing the relevance of retrieved documents to a user's question.\\n\\n"
    "Retrieved context (summarize mentally from the text):\\n{context}\\n\\n"
    "User's question:\\n{question}\\n\\n"
    "If the context likely contains keywords, ideas, or semantic meaning related to the question, "
    "respond only with 'yes'. Otherwise, respond only with 'no'."
)

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Return 'yes' if relevant, else 'no'.")

def _collect_latest_tool_context_and_question(state: MessagesState):
    # Last human question:
    question = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "human":
            question = msg.content
            break

    # Latest tool message (content + artifacts)
    context_text = ""
    artifacts = []
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "tool":
            context_text = msg.content or ""
            # Some runtimes attach artifacts on the message, others embed in additional kwargs
            artifacts = getattr(msg, "artifact", []) or getattr(msg, "artifacts", []) or []
            break
    return question, context_text, artifacts

def grade_documents(state: MessagesState) -> Literal["generate", "rewrite"]:
    question, context_text, _ = _collect_latest_tool_context_and_question(state)

    prompt = GRADE_PROMPT.format(question=question, context=context_text)
    response = llm_grader.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])
    return "generate" if response.binary_score.strip().lower() == "yes" else "rewrite"

# ------------------ Rewrite (clarify/improve the question) ------------------
REWRITE_PROMPT = (
    "Look at the input and infer the underlying intent.\\n"
    "Rewrite the question to be clearer and more specific for retrieval.\\n"
    "Original question:\\n"
    "-------\\n{question}\\n-------\\n"
    "Return only the improved question."
)

def rewrite(state: MessagesState):
    # Take last human message as the current question
    question = ""
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", "") == "human":
            question = msg.content
            break
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm_router.invoke([{"role": "user", "content": prompt}])
    # Push a new user-style message so router will try tools again
    return {"messages": [{"role": "user", "content": response.content}]}

# ------------------ Generate (final answer without chain-of-thought) ------------------
SYSTEM_POLICY = (
    "You are a professional Scrum Master specializing in Agile Project Management in Software Engineering.\\n"
    "When answering:\\n"
    "- Select ONLY the fields relevant to the question from {user story, description, acceptance criteria, tasks, effort, priority, ...}.\\n"
    "- For each selected field, give a one-line reason why it matters.\\n"
    "- Provide a brief rationale (1–2 sentences). Do NOT reveal step-by-step chain-of-thought.\\n"
    "- Cite 1–2 short snippets (<=20 words) from the retrieved context as evidence, include US-ID when available.\\n"
    "- Then give a concise final answer.\\n"
    "- If information is insufficient, say 'insufficient evidence' and ask for more documents."
)

def _tool_messages_text(state: MessagesState) -> str:
    """Concatenate recent ToolMessages' content for evidence-aware generation."""
    recent = []
    for m in reversed(state["messages"]):
        if getattr(m, "type", "") == "tool":
            recent.append(m.content or "")
        else:
            break
    return "\\n\\n".join(reversed(recent))

def generate(state: MessagesState):
    docs_content = _tool_messages_text(state)
    # Retain only human/system/ai (no tool-call ai) messages for dialog continuity
    conversation_messages = [
        m for m in state["messages"]
        if m.type in ("human", "system") or (m.type == "ai" and not getattr(m, "tool_calls", None))
    ]

    sys = SystemMessage(SYSTEM_POLICY + "\\n\\nRetrieved context follows:\\n" + docs_content)
    response = llm_answer.invoke([sys] + conversation_messages)
    return {"messages": [response]}

# ------------------ Build the graph ------------------
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("router", query_or_respond)
graph_builder.add_node("tools", tools_node)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)

graph_builder.set_entry_point("router")

# After router: if it called a tool, go to tools; otherwise, end (rare).
graph_builder.add_conditional_edges(
    "router",
    tools_condition,
    {"tools": "tools", END: END},
)

# After tools: grade → either generate or rewrite
graph_builder.add_conditional_edges("tools", grade_documents)
graph_builder.add_edge("generate", END)
graph_builder.add_edge("rewrite", "router")

# ------------------ Compile with persistent memory ------------------
with SqliteSaver.from_conn_string("rag_state.sqlite") as saver:
    graph = graph_builder.compile(checkpointer=saver)

    # ------------------ Streamlit Chat UI ------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Render chat history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Ask your Question")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Full history to the graph
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        ai_text = None
        placeholder = st.chat_message("assistant").empty()
        for step in graph.stream({"messages": history}, stream_mode="values", config=config):
            msgs = step.get("messages", [])
            if msgs:
                last = msgs[-1]
                if getattr(last, "type", None) == "ai" and getattr(last, "content", ""):
                    ai_text = last.content
                    placeholder.markdown(ai_text)

        if ai_text:
            st.session_state.messages.append({"role": "assistant", "content": ai_text})

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.experimental_rerun()
