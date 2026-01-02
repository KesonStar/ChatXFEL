from onnx import ModelProto
import streamlit as st
import sys
import time
import json
import uuid
from datetime import datetime
from pathlib import Path
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.chat_models import ChatOllama
from streamlit import session_state as ss
from streamlit.runtime.scriptrunner import get_script_run_ctx
sys.path.append('/home/zhangxf2/LLM/llm-shine/ChatXFEL')
import rag, utils
from research_agent import DeepResearchAgent

# App title
#st.set_page_config(page_title="ChatXFEL", layout='wide')
st.set_page_config(page_title="ChatXFEL Beta 1.0", page_icon='./draw/logo.png')

def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Fraunces:opsz,wght@9..144,600;700&display=swap');
        :root {
            --bg: #f6f4ef;
            --bg-accent: #efe8d9;
            --panel: #fffaf3;
            --panel-2: #f4eee4;
            --border: rgba(53, 45, 32, 0.12);
            --text: #1f1b16;
            --muted: #6a6355;
            --accent: #d46a4a;
            --accent-2: #2f6d62;
            --shadow: 0 10px 30px rgba(40, 32, 20, 0.12);
        }
        html, body, [class*="stApp"] {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
        }
        div[data-testid="stAppViewContainer"] {
            background: radial-gradient(1200px 600px at 10% -10%, #fff6e6 0%, transparent 60%),
                        radial-gradient(1200px 600px at 110% 10%, #e6f1ee 0%, transparent 55%),
                        var(--bg);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        #MainMenu,
        footer {
            visibility: hidden;
        }
        div.block-container {
            max-width: 900px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, h4 {
            font-family: 'Fraunces', serif;
            letter-spacing: -0.02em;
        }
        .hero {
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.4rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
        }
        .hero-subtitle {
            color: var(--muted);
            margin: 0 0 0.9rem 0;
            font-size: 1rem;
        }
        .hero-badges span {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: #f3e6d1;
            border: 1px solid var(--border);
            color: #513f2b;
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.8rem;
            margin-right: 0.45rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f9f4ea 0%, #f1ebe0 100%);
            border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: 'Fraunces', serif;
        }
        section[data-testid="stSidebar"] .stButton > button {
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #f5e8d5;
            color: #4a3a28;
        }
        div[data-testid="stChatMessage"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.35rem 0.85rem;
            box-shadow: 0 6px 20px rgba(40, 32, 20, 0.08);
            margin-bottom: 0.75rem;
        }
        div[data-testid="stChatMessage"][data-message-author-role="user"] {
            background: #eff6f4;
            border-color: rgba(47, 109, 98, 0.3);
        }
        div[data-testid="stChatMessageContent"] p {
            font-size: 1rem;
            line-height: 1.65;
        }
        div[data-testid="stChatInput"] textarea {
            border-radius: 16px;
            border: 1px solid var(--border);
            background: #fffdf9;
            padding: 0.8rem 1rem;
            box-shadow: 0 8px 20px rgba(40, 32, 20, 0.08);
        }
        div[data-testid="stChatInput"] textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(212, 106, 74, 0.2);
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">ChatXFEL</div>
        <div class="hero-subtitle">XFEL literature Q&amp;A with grounded citations and research modes.</div>
        <div class="hero-badges">
            <span>Beta</span>
            <span>Research-ready</span>
            <span>RAG-powered</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if 'agree' not in ss:
    ss['agree'] = False
def update_agree():
    ss['agree'] = True
    
if not ss['agree']:
    with st.empty():
        msg = '''This page is an intelligent system to answer the questions in the field of XFEL. If you click the **agree box** below, 
        :red[you IP and the time will be recorded]. If you don't agree with that, please close the page. 
        This note will appear again when you refresh the page.''' 
        st.markdown(msg)
    agree = st.checkbox('Agree', key='read', value=False, on_change=update_agree)
    while True:
        time.sleep(3600)

def reset_retriever_cache():
    try:
        get_retriever.clear()
        get_retriever_runtime.clear()
    except Exception as e:
        pass

initial_message = {"role": "assistant", "content": "What do you want to know about XFEL?"}

HISTORY_FILE = Path(__file__).with_name(".chatxfel_history.json")
HISTORY_VERSION = 1

def load_conversation_store():
    default_store = {"version": HISTORY_VERSION, "conversations": []}
    if not HISTORY_FILE.exists():
        return default_store
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return default_store
        if not isinstance(data.get("conversations"), list):
            return default_store
        data.setdefault("version", HISTORY_VERSION)
        return data
    except Exception as exc:
        print(f"Failed to load conversation history: {exc}")
        return default_store

def save_conversation_store(store):
    try:
        with HISTORY_FILE.open("w", encoding="utf-8") as handle:
            json.dump(store, handle, ensure_ascii=True, indent=2)
    except Exception as exc:
        print(f"Failed to save conversation history: {exc}")

def get_conversation(store, convo_id):
    for convo in store.get("conversations", []):
        if convo.get("id") == convo_id:
            return convo
    return None

def upsert_conversation(store, convo):
    for idx, existing in enumerate(store.get("conversations", [])):
        if existing.get("id") == convo.get("id"):
            store["conversations"][idx] = convo
            return
    store["conversations"].append(convo)

def infer_conversation_title(messages, fallback="New chat"):
    for item in messages:
        if item.get("role") == "user":
            content = str(item.get("content", "")).strip()
            if content:
                compact = " ".join(content.split())
                if len(compact) > 40:
                    compact = compact[:40].rstrip() + "..."
                return compact
    return fallback

def make_new_conversation():
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "id": uuid.uuid4().hex[:12],
        "title": "New chat",
        "created_at": now,
        "updated_at": now,
        "messages": [initial_message],
    }

def activate_conversation(convo):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    ss.active_conversation_id = convo.get("id")
    ss.active_conversation_created_at = convo.get("created_at", now)
    ss.messages = convo.get("messages", [initial_message])

def persist_current_conversation():
    if "conversation_store" not in ss or "active_conversation_id" not in ss or "messages" not in ss:
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    title = infer_conversation_title(ss.messages)
    created_at = ss.get("active_conversation_created_at", now)
    convo = {
        "id": ss.active_conversation_id,
        "title": title,
        "created_at": created_at,
        "updated_at": now,
        "messages": [dict(item) for item in ss.messages],
    }
    upsert_conversation(ss.conversation_store, convo)
    save_conversation_store(ss.conversation_store)

def delete_conversation(convo_id):
    store = ss.conversation_store
    remaining = [item for item in store.get("conversations", []) if item.get("id") != convo_id]
    store["conversations"] = remaining
    save_conversation_store(store)
    if remaining:
        latest = sorted(remaining, key=lambda item: item.get("updated_at", ""), reverse=True)[0]
        activate_conversation(latest)
    else:
        convo = make_new_conversation()
        upsert_conversation(store, convo)
        save_conversation_store(store)
        activate_conversation(convo)

# Initialize conversation store and active session
if "conversation_store" not in ss:
    ss.conversation_store = load_conversation_store()

if "active_conversation_id" not in ss:
    existing = ss.conversation_store.get("conversations", [])
    if existing:
        latest = sorted(existing, key=lambda item: item.get("updated_at", ""), reverse=True)[0]
        activate_conversation(latest)
    else:
        convo = make_new_conversation()
        upsert_conversation(ss.conversation_store, convo)
        save_conversation_store(ss.conversation_store)
        activate_conversation(convo)
elif "messages" not in ss:
    convo = get_conversation(ss.conversation_store, ss.active_conversation_id)
    if convo is None:
        convo = make_new_conversation()
        upsert_conversation(ss.conversation_store, convo)
        save_conversation_store(ss.conversation_store)
    activate_conversation(convo)

with st.sidebar:
    st.title('ChatXFEL Beta 2.0')
    #st.markdown('[About ChatXFEL](https://confluence.cts.shanghaitech.edu.cn/pages/viewpage.action?pageId=129762874)')
    st.markdown('[ChatXFEL简介与提问技巧](https://confluence.cts.shanghaitech.edu.cn/pages/viewpage.action?pageId=129762874)')
    #st.write(':red[You have agreed the recording of your IP and access time.]')
    #st.markdown('**IMPORTANT: The answers given by ChatXFEL are for informational purposes only, please consult the references in the source.**')
    st.markdown('**重要提示：大模型的回答仅供参考，点击Sources查看参考文献**')

    st.sidebar.markdown("---")
    st.sidebar.subheader("Conversations")
    if st.sidebar.button("New chat", type="primary"):
        persist_current_conversation()
        convo = make_new_conversation()
        upsert_conversation(ss.conversation_store, convo)
        save_conversation_store(ss.conversation_store)
        activate_conversation(convo)
        st.rerun()

    conversations = ss.conversation_store.get("conversations", [])
    if conversations:
        sorted_convos = sorted(conversations, key=lambda item: item.get("updated_at", ""), reverse=True)
        id_map = {item.get("id"): item for item in sorted_convos}
        convo_ids = [item.get("id") for item in sorted_convos if item.get("id")]
        if convo_ids:
            for convo_id in convo_ids:
                convo = id_map.get(convo_id, {})
                title = convo.get("title", "New chat")
                is_active = convo_id == ss.active_conversation_id
                button_type = "primary" if is_active else "secondary"
                if st.sidebar.button(title, key=f"conv_{convo_id}", type=button_type):
                    if not is_active:
                        persist_current_conversation()
                        activate_conversation(convo)
                        st.rerun()
            active_meta = id_map.get(ss.active_conversation_id)
            if active_meta and active_meta.get("updated_at"):
                st.sidebar.caption(f"Last updated {active_meta.get('updated_at')}")
            delete_disabled = ss.active_conversation_id is None or len(convo_ids) == 0
            if st.sidebar.button("Delete chat", type="secondary", disabled=delete_disabled):
                delete_conversation(ss.active_conversation_id)
                st.rerun()
    else:
        st.sidebar.caption("No saved conversations yet.")

    # Mode selection: Basic RAG vs Deep Research
    st.sidebar.markdown("---")
    st.sidebar.subheader("Research Mode")
    research_mode = st.sidebar.radio(
        "Select mode:",
        ["Basic RAG", "Deep Research"],
        key="research_mode",
        help="Basic RAG: Quick Q&A | Deep Research: Structured literature review"
    )
    if research_mode == "Deep Research":
        with st.popover(':information_source: About Deep Research'):
            msg = '''**Deep Research** generates comprehensive literature reviews through:
            1. **Clarification**: Ask questions to understand your research needs
            2. **Knowledge Extraction**: Create structured outline of key topics
            3. **Parallel Search**: Retrieve relevant papers for each topic
            4. **Review Generation**: Synthesize findings into a structured review

            This mode takes longer but provides more comprehensive analysis.'''
            st.markdown(msg)
    st.sidebar.markdown("---")

    # Refactored from https://github.com/a16z-infra/llama2-chatbot
    #st.subheader('Models and parameters')
    #model_list = ['LLaMA3.1-8B', 'Qwen2.5-7B']
    model_list = ['Qwen3-30B-Instruct', 'Qwen3-30B-thinking']
    col_list = ['xfel_bibs_collection_with_abstract', 'chatxfel', 'report', 'book']
    embedding_list = ['BGE-M3']

    selected_model = st.sidebar.selectbox('LLM model', model_list, index=0, key='selected_model')
    n_recall = 6 if selected_model.startswith('Q') else 5
    #if selected_model == 'LLaMA3-8B':
    #    #model_path = '/data-10gb/data/llm/gguf/Meta-Llama-3-8B-Instruct-Q8_0.gguf'
    #    n_recall = 5
    #elif selected_model == 'LLaMA3.1-8B':
    #    #model_path = '/data-10gb/data/llm/gguf/Meta-Llama-3-8B-Instruct-Q8_0.gguf'
    #    n_recall = 5
    #elif selected_model == 'Qwen2.5-7B':
    #    #model_path = '/data'
    #    n_recall = 5
    #elif selected_model == 'Qwen2.5-14B':
    #    #model_path = '/data-10gb/data/llm/qwen/qwen2-7b-instruct-q8_0.gguf'
    #    n_recall = 5

    selected_em = st.sidebar.selectbox('Embedding model', embedding_list, key='selected_em')
    if selected_em == 'llama2-7b':
        col_list.append('llama2_7b')
    elif selected_em == 'llama3-8b':
        col_list.append('llama3_8b')
    selected_col = st.sidebar.selectbox('Bibliography collection', col_list, key='select_col', on_change=reset_retriever_cache)
    col_name = selected_col
    with st.popover('About the collection'):
        if col_name == 'xfel_bibs_collection_with_abstract':
            msg = '''This is your custom XFEL bibliography collection with abstracts.'''
            st.markdown(msg)
        if col_name == 'book':
            msg = '''This collection now only contains some theses from EuXFEL.'''
            st.markdown(msg)
        if col_name == 'chatxfel':
            msg = '''This collection contains 3000+ publications of wordwide XFEL facilities, so ChatXFEL may be slower than other collections'''
            st.markdown(msg)
        if col_name == 'report':
            msg = '''This collection only contains unpulished references, e.g CDR, TDR, engineering reports.'''
            st.markdown(msg)

    filter_year = st.sidebar.checkbox('Filter papers by year', key='filter_year', value=True)
    if filter_year:
        min_year = 1949
        max_year = datetime.now().year
        year1, year2 = st.columns([1,1])
        #year_start = year1.selectbox('Start', list(range(min_year, max_year+1))[::-1], key='year_start', index=max_year-min_year)
        year_start = year1.selectbox('Start', list(range(min_year, max_year+1))[::-1], key='year_start', index=max_year-2000)
        year_end = year2.selectbox('End', list(range(year_start, max_year+1))[::-1], key='year_end')
    filter_title = False
    #filter_title = st.sidebar.checkbox('Filter by keywords', key='filter_title', value=False)
    #if filter_title:
    #    keywords = []
    #    key_title = st.sidebar.text_input('Keywords in title', key='key_title', placeholder='seperate by comma')
    #    if key_title != '':
    #        words = key_title.split(',')
    #        for word in words:
    #            keywords.append(word.strip().lower())

    n_batch, n_ctx, max_tokens = 512, 8192, 8192
    #return_source = st.sidebar.checkbox('Return Source', key='source', value=True)
    return_source = True
    use_mongo = True
    enable_chat_history = st.sidebar.checkbox('Enable chat history', key='chat_history', value=True)
    if enable_chat_history:
        with st.popover(':information_source: About chat history'):
            msg = '''When enabled, the model will see previous conversation turns to provide context-aware responses.
            This helps with follow-up questions and references to previous answers.'''
            st.markdown(msg)

    # button to Enable Query Rewrite
    use_query_rewrite = st.sidebar.checkbox('Enable Query Rewrite', key='query_rewrite', value=True)

    # Hybrid Search settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Retrieval Mode")
    use_hybrid_search = st.sidebar.checkbox('Enable Hybrid Search', key='hybrid_search', value=False)
    if use_hybrid_search:
        with st.popover(':information_source: About Hybrid Search'):
            msg = '''Hybrid Search combines:
            - **Dense Vector**: Semantic similarity search (understands meaning)
            - **Sparse Vector**: Keyword matching (like traditional search)

            Adjust weights to balance between semantic understanding and exact keyword matches.
            Uses Reciprocal Rank Fusion (RRF) to merge results.'''
            st.markdown(msg)

        st.sidebar.markdown("**Search Weight Balance:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            dense_weight = st.slider('Dense (Semantic)', min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='dense_weight')
        with col2:
            sparse_weight = st.slider('Sparse (Keyword)', min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='sparse_weight')

        # Show normalized weights
        total = dense_weight + sparse_weight
        if total > 0:
            st.sidebar.caption(f"Normalized: Dense={dense_weight/total:.1%}, Sparse={sparse_weight/total:.1%}")
    else:
        dense_weight = 0.5
        sparse_weight = 0.5

    use_routing = st.sidebar.checkbox('Enable DOI Scoped Search', key='use_routing', value=False,
                                    help="First find relevant papers (DOI), then search strictly within those papers.")
    """
    当前仅在xfel_bibs_collection_with_abstract这一个库中做abstract与text的检索。不过理论上是可以在两个分开的向量库中检索的
    """

    st.sidebar.markdown("---")
    st.sidebar.subheader("Response Style")
    enable_streaming = st.sidebar.checkbox('Type-out response', key='stream_response', value=True)
    stream_speed = st.sidebar.slider(
        'Typing speed (chars/sec)',
        min_value=30,
        max_value=200,
        value=120,
        step=10,
        key='stream_speed',
        disabled=not enable_streaming
    )

    st.sidebar.markdown("---")
    enable_log = st.sidebar.checkbox('Enable log', key='log', value=True)
    use_monog = False
    if enable_log:
        with st.popover(':warning: :red[About the log]'):
            msg = '''All the questions, answers, retrieved documents, and the question time will be logged.
            The logs would only be used for the development of ChatXFEL. \n\nIf you don't like the log, just uncheck the box "Enable log" above.
            \n\n**Your IP address will always be recorded.**'''
            st.markdown(msg)

@st.cache_resource
def get_embedding(embedding_model, n_ctx, n_gpu_layers=1):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting embedding...")
    # Get embedding
    if embedding_model == 'BGE-M3':
        embedding = rag.get_embedding_bge()
    return embedding
embedding = get_embedding(embedding_model=selected_em, n_ctx=n_ctx)
#print(f'Embedding: {embedding}')

#@st.cache_resource
#def get_llm(model_name, model_path, n_batch, n_ctx, max_tokens):
#    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting llm...")
#    # Get llm
#    llm = rag.get_llm_llama(model_name=model_name, model_path=model_path, n_batch=n_batch,n_ctx=n_ctx,verbose=False,
#                            streaming=True,max_tokens=max_tokens, temperature=0.8)
#    return llm
#llm = get_llm(selected_model, model_path, n_batch, n_ctx, max_tokens)

#def get_llm_ollama(model_name, num_predict, num_ctx=8192, keep_alive=-1, temperature=0.1, base_url='http://10.15.85.78:11434'):
@st.cache_resource
def get_llm(model_name, num_predict, keep_alive, num_ctx=8192, temperature=0.8):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting LLM...")
    llm = rag.get_llm_ollama(model_name=model_name, num_predict=num_predict, 
                             keep_alive=keep_alive, num_ctx=num_ctx, temperature=temperature, base_url='http://10.15.102.186:9000')
    return llm
llm = get_llm(model_name=selected_model, num_predict=2048, keep_alive=-1)

#You should answer the question in detail as far as possible. Do not make up questions by yourself.
#If you cannot find anwser in the context, just say that you don't know, don't try to make up an answer.
#Please remember some common abbrevations: SFX is short for serial femtosecond crystallography, SPI is
#short for single particle imaging.
#
#{context}
#
#Question: {question}
#Helpful Answer:"""

# Load appropriate prompt template based on chat history setting
if enable_chat_history:
    prompt_file = 'prompts/chat_with_history.pt'
else:
    prompt_file = 'prompts/naive.pt'

with open(prompt_file, 'r') as f:
    prompt_template = f.read()

@st.cache_data
def get_prompt_template(template):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting prompt...")
    prompt = rag.get_prompt(template)
    return prompt
prompt = get_prompt_template(template=prompt_template)

@st.cache_resource
def get_rerank_model(model_name='', top_n=n_recall):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting rerank model...")
    if model_name == '':
        model_name = 'BAAI/bge-reranker-v2-m3'
    rerank_model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)
    return compressor

connection_args = utils.get_milvus_connection(
    host='10.19.48.181',
    port=19530,
    username='cs286_2025_group8',
    password='Group8',
    db_name='cs286_2025_group8'
)
@st.cache_resource
def get_retriever(connection_args, col_name, _embedding):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting retriever...")
    if selected_em in ['llama2-7b', 'llama3-8b']:
        retriever = rag.get_retriever(connection_args=connection_args, col_name=col_name,
                                      embedding=_embedding, use_rerank=False, return_as_retreiever=False)
    else:
        retriever = rag.get_retriever(connection_args=connection_args, col_name=col_name,
                                      embedding=_embedding, vector_field='dense_vector',
                                      use_rerank=False, return_as_retreiever=False)
    return retriever

@st.cache_resource
def get_embedding_bge_m3_cached():
    """
    Get BGE-M3 embedding function for hybrid search.

    Returns:
        BGE-M3 embedding function or None if loading fails
    """
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: loading BGE-M3 embedding for hybrid search...")
    try:
        return rag.get_embedding_bge_m3()
    except Exception as e:
        st.error(f"Failed to load BGE-M3 model for hybrid search: {e}")
        st.warning("Hybrid search unavailable. Please use dense-only search mode or check model availability.")
        return None

@st.cache_resource
def get_hybrid_retriever_cached(connection_args, col_name, _embedding_m3, dense_w, sparse_w, top_n, filters=None):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: getting hybrid retriever...")
    retriever = rag.get_hybrid_retriever(
        connection_args=connection_args,
        col_name=col_name,
        embedding=_embedding_m3,
        dense_weight=dense_w,
        sparse_weight=sparse_w,
        use_rerank=True,
        top_n=top_n,
        filters=filters
    )
    return retriever

@st.cache_resource
def get_retriever_runtime(_retriever_obj, _compressor, filters=None):
    """
    Wrap a retriever object with a compressor.
    Handles both Milvus vector stores (need .as_retriever()) and BaseRetriever instances.
    """
    from langchain_core.retrievers import BaseRetriever

    # Check if the object is already a retriever
    if isinstance(_retriever_obj, BaseRetriever):
        # Already a retriever, use it directly
        base_retriever = _retriever_obj
    else:
        # It's a vector store (e.g., Milvus), convert to retriever
        search_kwargs = {'k': 10}
        if filters:
            search_kwargs = {**search_kwargs, **filters}
        base_retriever = _retriever_obj.as_retriever(search_kwargs=search_kwargs)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_compressor,
        base_retriever=base_retriever
    )
    return compression_retriever

# Prepare filters
filters = {}
if filter_year:
    #filters['expr'] = f'year >= {year_start} and year <= {year_end}'
    filters['expr'] = f'{year_start} <= year <= {year_end}'

# """
# Retriever Architecture:

# There are two retrieval modes:

# 1. Dense-only Search (default):
#    - Uses HuggingFaceBgeEmbeddings for semantic search
#    - Single dense vector (1024-dim) stored in 'dense_vector' field
#    - Returns: Milvus vector store -> wrapped with reranker -> ContextualCompressionRetriever

# 2. Hybrid Search (when enabled in sidebar):
#    - Uses BGEM3EmbeddingFunction for dual-vector search
#    - Dense vector (1024-dim) + Sparse vector (keyword-based)
#    - Combines results using Reciprocal Rank Fusion (RRF)
#    - Returns: HybridRetriever (BaseRetriever) -> optionally wrapped with reranker
#    - Falls back to dense-only if BGE-M3 model fails to load

# Both paths produce a retriever that can be used with rag.retrieve_generate()
# """

# Create retriever based on mode selection
if use_hybrid_search:
    # Use hybrid search mode with reranking
    # Get BGE-M3 embedding function (required for hybrid search with dense + sparse vectors)
    embedding_m3 = get_embedding_bge_m3_cached()

    if embedding_m3 is None:
        # Fall back to dense-only search if BGE-M3 fails to load
        st.warning(" Falling back to dense-only search mode due to BGE-M3 loading error.")
        retriever_obj = get_retriever(connection_args, selected_col, embedding)
        compressor = get_rerank_model(top_n=n_recall)
        retriever = get_retriever_runtime(retriever_obj, compressor, filters=filters)
    else:
        # BGE-M3 loaded successfully, use hybrid search
        filter_expr = filters.get('expr', None)
        retriever = get_hybrid_retriever_cached(
            connection_args, selected_col, embedding_m3,
            dense_weight, sparse_weight, n_recall, filters=filter_expr
        )
else:
    # Use traditional dense-only search
    retriever_obj = get_retriever(connection_args, selected_col, embedding)
    compressor = get_rerank_model(top_n=n_recall)
    retriever = get_retriever_runtime(retriever_obj, compressor, filters=filters)

# 应用路由逻辑
if use_routing:
    print(f"Initializing Routing...")
    
    # 确定 Embedding
    routing_embedding = embedding_m3 if (use_hybrid_search and embedding_m3) else embedding
    
    retriever = rag.get_routing_retriever(
        connection_args=connection_args,
        abstract_retriever=retriever,   # Step 1: 广撒网检索
        fulltext_col_name=selected_col, # Step 2: 在指定库中（这里是自己）按DOI检索
        embedding_function=routing_embedding,
        fulltext_top_k=n_recall 
    )

#if filter_title:
#    expr_title = ''
#    for i, word in enumerate(keywords):
#        if i ==  len(keywords) -1:
#            expr_title += f'\"{word}\" in title'
#        else:
#            expr_title += f'\"{word}\" in title and '
#    if 'expr' in filters.keys():
#        filters['expr'] += ' and ' + expr_title
#    else:
#        filters['expr'] = expr_title

# Deep Research state initialization
if "dr_stage" not in ss:
    ss.dr_stage = "initial"  # initial, clarification, confirmation, searching, review
if "dr_original_question" not in ss:
    ss.dr_original_question = ""
if "dr_clarification_questions" not in ss:
    ss.dr_clarification_questions = None
if "dr_clarifications" not in ss:
    ss.dr_clarifications = {}
if "dr_knowledge_outline" not in ss:
    ss.dr_knowledge_outline = None
if "dr_search_results" not in ss:
    ss.dr_search_results = None
if "dr_final_review" not in ss:
    ss.dr_final_review = ""
if "dr_references" not in ss:
    ss.dr_references = ""
if "dr_review_streamed" not in ss:
    ss.dr_review_streamed = False

def reset_deep_research():
    """Reset all Deep Research state variables."""
    ss.dr_stage = "initial"
    ss.dr_original_question = ""
    ss.dr_clarification_questions = None
    ss.dr_clarifications = {}
    ss.dr_knowledge_outline = None
    ss.dr_search_results = None
    ss.dr_final_review = ""
    ss.dr_references = ""
    ss.dr_review_streamed = False

@st.cache_resource
def get_deep_research_agent(_llm, _retriever, _reranker=None):
    """Create and cache the Deep Research Agent."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: initializing Deep Research Agent...")
    return DeepResearchAgent(_llm, _retriever, _reranker)

def log_feedback(feedback:dict, use_mongo):
    if feedback.get('Feedback', '') == '':
        feedback['Feedback'] = ss['feedback']+1
    utils.log_rag(feedback, use_mongo=use_mongo)

for message in ss.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("rewritten_query"):
            with st.expander("Optimized Search Query", expanded=False):
                st.markdown("**Rewritten for Search:**")
                st.success(message["rewritten_query"])
        #try:
        c = st.columns([8,2.5])
        if 'source' in message.keys():
            with c[0].popover('Sources'):
                st.markdown(message['source'])
            if message == ss.messages[-1]:
                if 'feedback' in ss:
                    ss['feedback'] = None
                with c[1]:
                    feedback = st.feedback('stars', key='feedback', on_change=log_feedback, args=({'Feedback':''}, use_mongo,))
                    #if feedback is not None:
                    #    log_feedback({'Feedback':str(feedback+1)}, use_mongo=use_mongo)
                #with c[1]:
                #    good = st.button(':thumbsup:', key='feedback_good_1', on_click=log_feedback, args=({'Feedback':'Good'},use_mongo,))
                #with c[2]:
                #    bad = st.button(':thumbsdown:', key='feedback_bad_1', on_click=log_feedback, args=({'Feedback':'Bad'}, use_mongo,))
        #except Exception as e:
        #    pass
        #num += 1
        #if 'messages' in ss.keys():
        #    ele = st.columns([3,1,1,1,1])
        #    if 'source' in message.keys():
        #        with ele[0]:
        #            with st.popover('Show source'):
        #                st.write(message['source'])
        #        if message['role'] == 'assistant':
        #            ele[1].button('Like', key=f'01{num}')
        #            ele[2].button('Dislike', key=f'02{num}')
        #            ele[3].button('Retry', key=f'03{num}')
        #            ele[4].button('Modify', key=f'04{num}')

def stream_markdown(text, placeholder, chars_per_sec):
    if not text:
        placeholder.markdown("")
        return
    delay = 1.0 / max(chars_per_sec, 1)
    rendered = ""
    for char in text:
        rendered += char
        placeholder.markdown(rendered)
        if delay:
            time.sleep(delay)

def render_response(text, placeholder, stream, chars_per_sec):
    if stream:
        stream_markdown(text, placeholder, chars_per_sec)
    else:
        placeholder.markdown(text)

# Function for generating LLaMA2 response
def generate_llama2_response(question, use_history=False):
    # Format chat history if enabled
    history_text = ""
    if use_history and len(ss.messages) > 1:
        # Build chat history from previous messages (excluding initial message and current question)
        for dict_message in ss.messages[1:-1]:  # Skip initial greeting and current user question
            if dict_message["role"] == "user":
                history_text += "User: " + dict_message["content"] + "\n"
            elif dict_message["role"] == "assistant":
                # Get content without source references
                content = dict_message["content"]
                history_text += "Assistant: " + content + "\n"

        if history_text:
            history_text = history_text.strip()
        else:
            history_text = "No previous conversation."
    else:
        history_text = "No previous conversation."

    # Call RAG with or without history
    raw_question = question  # keep the user query clean for rewrite/retrieval
    if use_history:
        output = rag.retrieve_generate(
            question=raw_question,
            llm=llm,
            prompt=prompt,
            retriever=retriever,
            history=history_text,
            return_source=return_source,
            return_chain=False,
            use_query_rewrite=use_query_rewrite
        )
    else:
        output = rag.retrieve_generate(
            question=raw_question,
            llm=llm,
            prompt=prompt,
            retriever=retriever,
            return_source=return_source,
            return_chain=False,
            use_query_rewrite=use_query_rewrite
        )

    return output

@st.cache_data
def log_ip_time(session_id):
    ip = session.request.remote_ip
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {ip} connected or refreshed!", flush=True)

ctx = get_script_run_ctx()
client_ip = ''
if ctx:
    session = st.runtime.get_instance().get_client(ctx.session_id)
    client_ip = session.request.remote_ip
    log_ip_time(ctx.session_id)

# ============================================================
# BASIC RAG MODE
# ============================================================
if research_mode == "Basic RAG":
    # User-provided prompt
    question_time = ''
    if question:= st.chat_input():
        if enable_log:
            question_time = time.strftime('%Y-%m-%d %H:%M:%S')
        ss.messages.append({"role": "user", "content": question})
        persist_current_conversation()
        with st.chat_message("user"):
            st.write(question)

    # Generate a new response if last message is not from assistant
    if 'feedback_good' not in ss:
        ss['feedback_good'] = None
    if 'feedback_bad' not in ss:
        ss['feedback_bad'] = None

    if ss.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(question, use_history=enable_chat_history)

            # Display rewritten query if available
            if 'rewritten_query' in response and response['rewritten_query']:
                # Get the original user question (latest user turn)
                original_user_question = ss.messages[-1]["content"]
                with st.expander("Optimized Search Query", expanded=False):
                    st.markdown("**Original Question:**")
                    st.info(original_user_question)
                    st.markdown("**Rewritten for Search:**")
                    st.success(response['rewritten_query'])

            placeholder = st.empty()
            full_response = ''
            source = ''
            if return_source:
                full_response = response['answer']
                render_response(full_response, placeholder, enable_streaming, stream_speed)
                #full_response += '\nContext: \n'
                for i, c in enumerate(response['context']):
                    source += f'{c.page_content}'
                    title = c.metadata.get('title') if 'title' in c.metadata.keys() else c.metadata.get('source')
                    doi = c.metadata.get('doi', '')
                    journal = c.metadata.get('journal', '')
                    year = c.metadata.get('year', '')
                    page = c.metadata.get('page')
                    if doi == '':
                        source += f'\n\n**Ref. {i+1}**: {title}, {journal}, {year}, page {page}'
                    else:
                        source += f'\n\n**Ref. {i+1}**: {title}, {journal}, {year}, [{doi}](http://dx.doi.org/{doi}), page {page}'
                    if i != len(response['context'])-1:
                        source += '\n\n'
                    #placeholder.markdown(source)
                    if i == len(response['context'])-1:
                        c = st.columns([8,3])
                        with c[0].popover('Source'):
                            st.markdown(source)
            else:
                full_response = response.content
                render_response(full_response, placeholder, enable_streaming, stream_speed)

        if return_source:
            message = {"role": "assistant", "content": full_response, "source":source}
        else:
            message = {"role": "assistant", "content": full_response}

        if 'rewritten_query' in response and response['rewritten_query']:
            message["rewritten_query"] = response['rewritten_query']
        if enable_log:
            logs = {'IP':client_ip, 'Time':question_time, 'Model':selected_model, 'Question': question, 'Answer':full_response, 'Source':source}
            utils.log_rag(logs, use_mongo=use_mongo)
        ss.messages.append(message)
        persist_current_conversation()
        st.rerun()

# ============================================================
# DEEP RESEARCH MODE
# ============================================================
else:
    # Initialize Deep Research Agent
    dr_agent = get_deep_research_agent(llm, retriever, compressor)

    # Display current stage indicator
    stage_labels = {
        "initial": "1️⃣ Enter Research Topic",
        "clarification": "2️⃣ Answer Clarification Questions",
        "confirmation": "3️⃣ Confirm Knowledge Outline",
        "searching": "4️⃣ Searching Documents...",
        "review": "5️⃣ Literature Review Generated"
    }
    st.info(f"**Current Stage**: {stage_labels.get(ss.dr_stage, 'Unknown')}")

    # Stage 1: Initial - Enter research topic
    if ss.dr_stage == "initial":
        st.markdown("### Enter Your Research Topic")
        st.markdown("Describe the topic you want to explore. Be as specific as possible for better results.")

        research_topic = st.text_area(
            "Research Topic",
            placeholder="e.g., Recent advances in serial femtosecond crystallography data processing methods",
            height=100,
            key="dr_input_topic"
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Start Research", type="primary", disabled=not research_topic):
                ss.dr_original_question = research_topic
                with st.spinner("Generating clarification questions..."):
                    try:
                        ss.dr_clarification_questions = dr_agent.generate_clarification_questions(research_topic)
                        ss.dr_stage = "clarification"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating questions: {e}")

        with col2:
            # Quick mode: skip clarification
            if st.button("Quick Mode (Skip Clarification)", disabled=not research_topic):
                ss.dr_original_question = research_topic
                with st.spinner("Generating knowledge outline..."):
                    try:
                        ss.dr_knowledge_outline = dr_agent.extract_knowledge_points(research_topic, {})
                        ss.dr_stage = "confirmation"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating outline: {e}")

    # Stage 2: Clarification - Answer questions
    elif ss.dr_stage == "clarification":
        st.markdown("### Clarification Questions")
        st.markdown("Please answer the following questions to help refine the research scope:")

        st.info(f"**Your Research Topic**: {ss.dr_original_question}")

        questions = ss.dr_clarification_questions.get("questions", [])

        if questions:
            for q in questions:
                q_id = str(q.get("id", 0))
                question_text = q.get("question", "")
                purpose = q.get("purpose", "")

                # Initialize answer if not exists
                if q_id not in ss.dr_clarifications:
                    ss.dr_clarifications[q_id] = ""

                st.markdown(f"**Q{q_id}**: {question_text}")
                st.caption(f"Purpose: {purpose}")
                ss.dr_clarifications[q_id] = st.text_area(
                    f"Your answer for Q{q_id}",
                    value=ss.dr_clarifications.get(q_id, ""),
                    key=f"dr_clarification_{q_id}",
                    height=80,
                    label_visibility="collapsed"
                )
                st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back", key="dr_back_to_initial"):
                ss.dr_stage = "initial"
                st.rerun()
        with col2:
            if st.button("Generate Outline ➡️", type="primary"):
                with st.spinner("Generating knowledge outline..."):
                    try:
                        ss.dr_knowledge_outline = dr_agent.extract_knowledge_points(
                            ss.dr_original_question,
                            ss.dr_clarifications
                        )
                        ss.dr_stage = "confirmation"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating outline: {e}")

    # Stage 3: Confirmation - Review and edit outline
    elif ss.dr_stage == "confirmation":
        st.markdown("### Knowledge Point Outline")
        st.markdown("Review the generated outline. You can edit it before proceeding.")

        if ss.dr_knowledge_outline:
            # Display title
            st.subheader(ss.dr_knowledge_outline.get("title", "Literature Review"))

            # Display knowledge points
            st.markdown("**Knowledge Points:**")
            kps = ss.dr_knowledge_outline.get("knowledge_points", [])
            for kp in kps:
                importance_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(kp.get("importance", "medium"), "⚪")
                with st.expander(f"{importance_emoji} [{kp.get('category', 'Unknown')}] {kp.get('topic', 'Unknown')}", expanded=False):
                    st.markdown(f"**ID**: {kp.get('id', 'N/A')}")
                    st.markdown(f"**Search Keywords**: {', '.join(kp.get('search_keywords', []))}")
                    st.markdown(f"**Importance**: {kp.get('importance', 'medium')}")

            # Display search strategy
            st.markdown(f"**Search Strategy**: {ss.dr_knowledge_outline.get('search_strategy', 'N/A')}")

            # Edit option
            with st.expander("📝 Edit Outline (JSON)", expanded=False):
                edited_json = st.text_area(
                    "Edit JSON",
                    value=json.dumps(ss.dr_knowledge_outline, indent=2, ensure_ascii=False),
                    height=400,
                    key="dr_edit_outline"
                )
                if st.button("Apply Changes"):
                    try:
                        ss.dr_knowledge_outline = json.loads(edited_json)
                        st.success("Outline updated!")
                        st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back", key="dr_back_to_clarification"):
                ss.dr_stage = "clarification"
                st.rerun()
        with col2:
            if st.button("Start Search ➡️", type="primary"):
                ss.dr_stage = "searching"
                st.rerun()

    # Stage 4: Searching - Execute searches
    elif ss.dr_stage == "searching":
        st.markdown("### Searching Documents")
        st.markdown("Retrieving relevant documents for each knowledge point...")

        # Get year filter if enabled
        year_filter = None
        if filter_year:
            year_filter = (year_start, year_end)

        # Progress display
        kps = ss.dr_knowledge_outline.get("knowledge_points", [])
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Perform search
            status_text.text("Searching documents...")
            ss.dr_search_results = dr_agent.search_documents(ss.dr_knowledge_outline, year_filter)
            progress_bar.progress(50)

            # Display search summary
            status_text.text("Search complete. Generating review...")
            for kp_id, docs in ss.dr_search_results.items():
                st.write(f"- {kp_id}: {len(docs)} documents found")

            # Generate review
            progress_bar.progress(75)
            ss.dr_final_review = dr_agent.generate_review(ss.dr_knowledge_outline, ss.dr_search_results)
            ss.dr_review_streamed = False

            # Format references
            ss.dr_references = dr_agent.review_generator.format_references(ss.dr_search_results)

            progress_bar.progress(100)
            status_text.text("Review generated successfully!")

            ss.dr_stage = "review"
            st.rerun()

        except Exception as e:
            st.error(f"Error during search/generation: {e}")
            if st.button("Retry"):
                st.rerun()
            if st.button("⬅️ Back to Outline"):
                ss.dr_stage = "confirmation"
                st.rerun()

    # Stage 5: Review - Display final review
    elif ss.dr_stage == "review":
        st.markdown("### Literature Review Generated")

        # Display the review
        review_placeholder = st.empty()
        if enable_streaming and not ss.dr_review_streamed:
            render_response(ss.dr_final_review, review_placeholder, True, stream_speed)
            ss.dr_review_streamed = True
        else:
            review_placeholder.markdown(ss.dr_final_review)

        # References section
        with st.expander("📚 References", expanded=False):
            st.markdown(ss.dr_references)

        # Search results by knowledge point
        with st.expander("🔍 Documents by Knowledge Point", expanded=False):
            if ss.dr_search_results:
                for kp_id, docs in ss.dr_search_results.items():
                    st.markdown(f"**{kp_id}** ({len(docs)} documents)")
                    for doc in docs:
                        metadata = doc.metadata
                        title = metadata.get('title', 'Unknown')
                        year = metadata.get('year', 'N/A')
                        doi = metadata.get('doi', '')
                        if doi:
                            st.markdown(f"- {title} ({year}) - [DOI](https://doi.org/{doi})")
                        else:
                            st.markdown(f"- {title} ({year})")
                    st.markdown("---")

        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Research", type="primary"):
                reset_deep_research()
                st.rerun()
        with col2:
            # Download button
            st.download_button(
                label="📥 Download Review",
                data=ss.dr_final_review,
                file_name="literature_review.md",
                mime="text/markdown"
            )

    # Sidebar button to reset deep research
    with st.sidebar:
        if ss.dr_stage != "initial":
            st.sidebar.markdown("---")
            if st.sidebar.button("🔄 Reset Deep Research"):
                reset_deep_research()
                st.rerun()
