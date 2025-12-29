#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zhangxf2@shanghaitech.edu.cn
# Date: Mar, 29 2024

'''
Define the functions for RAG pipeline
'''

import os
import sys
from langchain_community.document_loaders import (PyPDFLoader, PDFPlumberLoader, 
        UnstructuredMarkdownLoader, BSHTMLLoader, JSONLoader, CSVLoader, DirectoryLoader)
from langchain_community.vectorstores import Milvus 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceBgeEmbeddings
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from transformers import AutoTokenizer
from langchain_community.chat_models import ChatOllama

from query_rewriter import rewrite_query
import torch
from pymilvus import Collection, connections
from typing import List, Dict, Any

# sys.path.append('/home/zhangxf/workdir/LLM/llm-shine/ChatXFEL/src')
import utils

def load(file_name, file_type='pdf', pdf_loader='pypdf'):
    '''
    load documents by following loader:
    pdf: PyPDFLoader or PDFPlumberLoader
    markdown: UnstructedMarkdownLoader
    html: BSHTMLLoader
    json: JSONLoader
    csv: CSVLoader
    
    Args:
        file_name: file name to be load
        file_type: pdf, markdown, html, json, csv
        loader: specify document loader
        split: load or load_and_split
    '''
    if not os.path.exists(file_name):
        print(f'ERROR: {file_name} does not exist')
        return []

    doc = []
    if file_type.lower() == 'pdf':
        if pdf_loader == 'pypdf':
            #loader = PyPDFLoader(file_name, extract_images=True)
            loader = PyPDFLoader(file_name)
            doc = loader.load()
        elif pdf_loader == 'pdfplumber':
            loader = PDFPlumberLoader(file_name)
            doc = loader.load()
        else:
            print('pdf_loader should be one of pypdf or pdfplumber')
    elif file_type.lower() == 'markdwon':
        loader = UnstructuredMarkdownLoader(file_name)
        doc = loader.load()
    elif file_type.lower() == 'html':
        loader = BSHTMLLoader(file_name)
        doc = loader.load()
    elif file_type.lower() == 'json':
        loader = JSONLoader(file_name)
        doc = loader.load()
    elif file_type.lower() == 'csv':
        loader = CSVLoader(file_name)
        doc = loader.load()
    else:
        print(f'Unsupported file type.')
        print('Supported file types are: pdf, markdown, html, json, csv')

    return doc

"""
The function has never been used in the project.
"""
# def load_pdf_directory(file_dir, recursive=True, multitread=True):
#     kwargs = {'extract_images':True}
#     loader = DirectoryLoader(file_dir, glob='**/*.pdf', loader_cls=PyPDFLoader, recursive=recursive,
#                              loader_kwargs=kwargs, show_progress=True, use_multithreading=multitread)
#     docs = loader.load()
#     return docs

def split(docs, size=2000, overlap=200, length_func=len, sep=None, is_regex=False): 
    '''
    only recursively split by character is used now.
    '''
    if type(docs) is not list:
        print(f'{docs} should be a list.')
        return []
    if sep != None:
        separator = sep
    else:
        separator = ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = size,
        chunk_overlap = overlap,
        length_function = length_func,
        is_separator_regex=is_regex,
        add_start_index = True,
        separators=separator)

    texts = splitter.split_documents(docs)
    return texts

def get_embedding_bge(model_kwargs=None, encode_kwargs=None):
    if model_kwargs is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        model_kwargs = {'device':device}
    if encode_kwargs is None:
        encode_kwargs = {'normalize_embeddings':True}
    model_name = 'BAAI/bge-m3'
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embedding


def get_embedding_bge_m3(device=None, use_fp16=False):
    """
    Get BGE-M3 embedding function for hybrid search (dense + sparse vectors).

    Args:
        device: Device to use ('cpu', 'cuda', or 'mps'). Auto-detected if None.
        use_fp16: Whether to use fp16 precision (only for CUDA)

    Returns:
        BGEM3EmbeddingFunction object with encode_documents method

    Raises:
        RuntimeError: If model fails to load
    """
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    model_name = 'BAAI/bge-m3'

    # fp16 only works with CUDA
    if use_fp16 and device != 'cuda':
        use_fp16 = False
        print(f"Warning: fp16 only supported on CUDA, using fp32 on {device}")

    try:
        embedding = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16
        )
        print(f"Successfully loaded BGE-M3 model on {device}")
        return embedding
    except Exception as e:
        error_msg = f"Failed to load BGE-M3 model '{model_name}' on device '{device}': {e}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


"""
The functions below are deprecated and bgem3 is used instead. 
"""
# def get_embedding(model_name, n_gpu_layers=-1, n_ctx=4096):
#     '''
#     Supported models: llama, gpt
#     '''
#     embedding = None
#     if 'llama' in model_name.lower():
#         embedding = LlamaCppEmbeddings(
#             model_path=model_name,
#             n_gpu_layers = n_gpu_layers,
#             n_ctx=n_ctx
#         )
#     elif 'gpt' in model_name.lower():
#         print('Support for GPT models is TBD')
#     else:
#         print('Only gpt or llama are supported')

#     return embedding

# def restore_vector(docs, connection_args, col_name, embedding, desc=''):
#     _ = Milvus(embedding_function=embedding,
#                           connection_args=connection_args,
#                           collection_name=col_name,
#                           drop_old=True
#                          ).from_documents(
#                              docs,
#                              embedding=embedding,
#                              connection_args=connection_args,
#                              collection_description=desc,
#                              collection_name=col_name
#                          )
#     return _

def get_retriever(connection_args, col_name, embedding, vector_field='vector', use_rerank=False,
                  top_n=4, filters=None, return_as_retreiever=True):
    search_kwargs = {'k':10, 'params': {'ef': 20}}
    if filters:
        search_kwargs['filter'] = filters
    retriever = Milvus(embedding_function=embedding,
                       connection_args=connection_args,
                       collection_name=col_name,
                       vector_field=vector_field)
    if use_rerank:
        rerank_model = HuggingFaceCrossEncoder(
            #model_name = '/data-10gb/data/llm/bge-reranker-v2-m3')
            model_name = 'BAAI/bge-reranker-v2-m3')
        compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                   base_retriever=retriever.as_retriever(search_kwargs=search_kwargs))
        return compression_retriever
    else:
        if return_as_retreiever:
            return retriever.as_retriever(search_kwargs=search_kwargs)
        else:
            return retriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines dense and sparse vector search using BGE-M3 embeddings.
    Uses Reciprocal Rank Fusion (RRF) to merge results from both searches.

    IMPORTANT: BGE-M3 uses different encoding strategies:
    - encode_documents(): For indexing documents (used in vectorize_bibs.py)
    - encode_queries(): For search queries (used in _get_relevant_documents)

    This distinction is crucial especially for sparse vectors, where query and document
    encoding have different keyword extraction strategies.
    """
    connection_args: Dict
    collection_name: str
    embedding_function: Any
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    top_k: int = 10
    filters: str = None
    collection: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, connection_args: Dict, collection_name: str, embedding_function,
                 dense_weight: float = 0.5, sparse_weight: float = 0.5,
                 top_k: int = 10, filters: str = None, **kwargs):
        """
        Args:
            connection_args: Milvus connection arguments
            collection_name: Name of the Milvus collection
            embedding_function: BGE-M3 embedding function
            dense_weight: Weight for dense vector search (0-1)
            sparse_weight: Weight for sparse vector search (0-1)
            top_k: Number of results to retrieve
            filters: Milvus filter expression
        """
        super().__init__(
            connection_args=connection_args,
            collection_name=collection_name,
            embedding_function=embedding_function,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=top_k,
            filters=filters,
            **kwargs
        )

        # Connect to Milvus with alias
        alias = f"hybrid_{collection_name}"
        try:
            # Check if connection already exists
            if alias not in connections.list_connections():
                # Prepare connection arguments
                conn_args = {
                    'alias': alias,
                    'host': connection_args.get('host', 'localhost'),
                    'port': connection_args.get('port', '19530')
                }
                if 'user' in connection_args:
                    conn_args['user'] = connection_args['user']
                if 'password' in connection_args:
                    conn_args['password'] = connection_args['password']
                if 'db_name' in connection_args:
                    conn_args['db_name'] = connection_args['db_name']
                if 'secure' in connection_args:
                    conn_args['secure'] = connection_args['secure']

                connections.connect(**conn_args)

            # Get collection
            self.collection = Collection(name=collection_name, using=alias)
            self.collection.load()
        except Exception as e:
            print(f"Error connecting to Milvus collection '{collection_name}': {e}")
            raise

    def _sparse_to_dict(self, sparse_embedding):
        """Convert sparse embedding to dict format for Milvus"""
        if hasattr(sparse_embedding, 'toarray'):
            # scipy sparse matrix (csr_matrix or similar)
            nonzero_indices = sparse_embedding.nonzero()

            # Handle both 1D and 2D sparse matrices
            if len(nonzero_indices) == 1:
                # 1D sparse array - only one set of indices
                indices = nonzero_indices[0]
            else:
                # 2D sparse matrix - use column indices
                indices = nonzero_indices[1]

            values = sparse_embedding.data
        elif isinstance(sparse_embedding, dict):
            # dict format
            indices = list(sparse_embedding.keys())
            values = list(sparse_embedding.values())
        else:
            # Assume it's already in the correct format (dict-like)
            # This handles the case where Milvus returns sparse vectors as dict
            return sparse_embedding

        return {int(idx): float(val) for idx, val in zip(indices, values)}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List:
        """
        Retrieve documents using hybrid search (dense + sparse vectors).

        Args:
            query: Search query string
            run_manager: Callback manager for retriever run

        Returns:
            List of Langchain Document objects

        Raises:
            RuntimeError: If embedding generation or search fails
        """
        from langchain_classic.schema.document import Document

        try:
            # Generate embeddings using BGE-M3
            # IMPORTANT: Use encode_queries for search queries, not encode_documents
            # BGE-M3 uses different encoding strategies for queries vs documents
            embeddings = self.embedding_function.encode_queries([query])
            dense_vec = embeddings['dense'][0]
            sparse_vec = embeddings['sparse'][0]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings for query '{query}': {e}") from e

        try:
            # Convert sparse embedding to dict format
            sparse_dict = self._sparse_to_dict(sparse_vec)
        except Exception as e:
            raise RuntimeError(f"Failed to convert sparse embedding to dict format: {e}") from e

        # Prepare search parameters
        search_params_dense = {"metric_type": "IP", "params": {"ef": 20}}
        search_params_sparse = {"metric_type": "IP"}

        try:
            # Perform dense vector search
            dense_results = self.collection.search(
                data=[dense_vec],
                anns_field="dense_vector",
                param=search_params_dense,
                limit=self.top_k,
                expr=self.filters,
                output_fields=['title', 'doi', 'journal', 'year', 'page', 'start_index', 'text']
            )
        except Exception as e:
            raise RuntimeError(f"Dense vector search failed: {e}") from e

        try:
            # Perform sparse vector search
            sparse_results = self.collection.search(
                data=[sparse_dict],
                anns_field="sparse_vector",
                param=search_params_sparse,
                limit=self.top_k,
                expr=self.filters,
                output_fields=['title', 'doi', 'journal', 'year', 'page', 'start_index', 'text']
            )
        except Exception as e:
            raise RuntimeError(f"Sparse vector search failed: {e}") from e

        # Apply Reciprocal Rank Fusion (RRF)
        rrf_k = 60  # RRF constant
        score_dict = {}

        # Process dense results
        for rank, hit in enumerate(dense_results[0]):
            doc_id = hit.id
            rrf_score = self.dense_weight / (rrf_k + rank + 1)
            if doc_id not in score_dict:
                score_dict[doc_id] = {
                    'score': rrf_score,
                    'entity': hit.entity
                }
            else:
                score_dict[doc_id]['score'] += rrf_score

        # Process sparse results
        for rank, hit in enumerate(sparse_results[0]):
            doc_id = hit.id
            rrf_score = self.sparse_weight / (rrf_k + rank + 1)
            if doc_id not in score_dict:
                score_dict[doc_id] = {
                    'score': rrf_score,
                    'entity': hit.entity
                }
            else:
                score_dict[doc_id]['score'] += rrf_score

        # Sort by combined score and take top_k
        sorted_results = sorted(score_dict.items(), key=lambda x: x[1]['score'], reverse=True)[:self.top_k]

        # Convert to Langchain Documents
        documents = []
        for doc_id, data in sorted_results:
            entity = data['entity']
            metadata = {
                'title': entity.get('title', ''),
                'doi': entity.get('doi', ''),
                'journal': entity.get('journal', ''),
                'year': entity.get('year', ''),
                'page': entity.get('page', ''),
                'start_index': entity.get('start_index', ''),
                'source': entity.get('title', ''),
                'hybrid_score': data['score']
            }
            doc = Document(
                page_content=entity.get('text', ''),
                metadata=metadata
            )
            documents.append(doc)

        return documents


def get_hybrid_retriever(connection_args, col_name, embedding,
                        dense_weight=0.5, sparse_weight=0.5,
                        use_rerank=False, top_n=4, filters=None):
    """
    Create a hybrid retriever that combines dense and sparse vector search.

    Args:
        connection_args: Milvus connection arguments
        col_name: Collection name
        embedding: BGE-M3 embedding function (must have encode_documents method)
        dense_weight: Weight for dense vector (default 0.5)
        sparse_weight: Weight for sparse vector (default 0.5)
        use_rerank: Whether to use reranker
        top_n: Number of final results after reranking
        filters: Milvus filter expression (e.g., "year >= 2020")

    Returns:
        HybridRetriever or ContextualCompressionRetriever

    Raises:
        ValueError: If embedding function is None or doesn't support encode_documents
        RuntimeError: If Milvus connection fails
    """
    # Validate embedding function
    if embedding is None:
        raise ValueError("Embedding function cannot be None for hybrid search")

    if not hasattr(embedding, 'encode_documents'):
        raise ValueError(
            f"Embedding function must have 'encode_documents' method for hybrid search. "
            f"Got {type(embedding).__name__}. Use get_embedding_bge_m3() instead."
        )

    # Normalize weights
    total_weight = dense_weight + sparse_weight
    if total_weight > 0:
        dense_weight = dense_weight / total_weight
        sparse_weight = sparse_weight / total_weight
    else:
        raise ValueError("At least one of dense_weight or sparse_weight must be > 0")

    try:
        retriever = HybridRetriever(
            connection_args=connection_args,
            collection_name=col_name,
            embedding_function=embedding,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=10,
            filters=filters
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create HybridRetriever: {e}") from e

    if use_rerank:
        try:
            rerank_model = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')
            compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)

            # HybridRetriever is now a BaseRetriever, so it can be used directly
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
            return compression_retriever
        except Exception as e:
            print(f"Warning: Failed to create reranker, using retriever without reranking: {e}")
            return retriever
    else:
        return retriever

# ... [保留前面的 imports 和 HybridRetriever 类] ...

class RoutingRetriever(BaseRetriever):
    """
    Two-stage retrieval implementation:
    Stage 1: Search in Base/Abstract Collection to find relevant DOIs.
    Stage 2: Search in Full-text Collection restricting scope to those DOIs.
    """
    abstract_retriever: BaseRetriever
    fulltext_collection: Any = None
    embedding_function: Any
    id_field: str = "doi"
    fulltext_top_k: int = 10
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, abstract_retriever, connection_args, fulltext_col_name, embedding_function, 
                 id_field="doi", fulltext_top_k=10, **kwargs):
        super().__init__(
            abstract_retriever=abstract_retriever,
            embedding_function=embedding_function,
            id_field=id_field,
            fulltext_top_k=fulltext_top_k,
            **kwargs
        )
        
        # Initialize connection to the Full Text Milvus Collection (even if it's the same collection)
        alias = f"routing_full_{fulltext_col_name}"
        try:
            if alias not in connections.list_connections():
                conn_args = {
                    'alias': alias,
                    'host': connection_args.get('host', 'localhost'),
                    'port': connection_args.get('port', '19530')
                }
                # Copy authentication args
                for key in ['user', 'password', 'db_name', 'secure']:
                    if key in connection_args:
                        conn_args[key] = connection_args[key]
                connections.connect(**conn_args)

            self.fulltext_collection = Collection(name=fulltext_col_name, using=alias)
            self.fulltext_collection.load()
            print(f"RoutingRetriever: Connected to target collection '{fulltext_col_name}'")
        except Exception as e:
            print(f"Error connecting to target collection '{fulltext_col_name}': {e}")
            raise

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List:
        from langchain_classic.schema.document import Document

        # --- Stage 1: Search Abstracts (Base Search) ---
        # print(f"Routing: Searching base layer for '{query}'...")
        abstract_docs = self.abstract_retriever.invoke(query)
        
        if not abstract_docs:
            return []

        # Extract unique IDs (DOIs)
        target_ids = list(set([
            doc.metadata.get(self.id_field) 
            for doc in abstract_docs 
            if doc.metadata.get(self.id_field)
        ]))

        if not target_ids:
            # print("Warning: No DOIs found via base retrieval. Returning base docs.")
            return abstract_docs

        # print(f"Routing: Found {len(target_ids)} relevant DOIs. Focusing search.")

        # --- Stage 2: Search Full Text with Filtering ---
        try:
            # 1. Prepare Milvus Expression
            ids_str = ", ".join([f"'{str(id_)}'" for id_ in target_ids])
            expr = f"{self.id_field} in [{ids_str}]"
            
            # 2. Embed the query
            if hasattr(self.embedding_function, 'encode_queries'):
                # BGE-M3
                embeddings_dict = self.embedding_function.encode_queries([query])
                dense_vec = embeddings_dict['dense'][0]
            elif hasattr(self.embedding_function, 'embed_query'):
                # Standard LangChain
                dense_vec = self.embedding_function.embed_query(query)
            else:
                # Fallback for old interfaces
                dense_vec = self.embedding_function.embed_documents([query])[0]

            # 3. Execute Search
            search_params = {"metric_type": "IP", "params": {"ef": 20}}
            
            # Add 'abstract' to output_fields just in case 'text' is empty
            results = self.fulltext_collection.search(
                data=[dense_vec],
                anns_field="dense_vector", 
                param=search_params,
                limit=self.fulltext_top_k,
                expr=expr, 
                output_fields=['title', 'doi', 'journal', 'year', 'page', 'text', 'abstract', 'source']
            )
            
            # 4. Convert to Documents
            documents = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity
                    
                    # Logic to determine page_content: Priority Text -> Fallback Abstract
                    content = entity.get('text', '')
                    if content is None or str(content).strip() == '':
                        content = entity.get('abstract', '')
                        
                    metadata = {
                        'title': entity.get('title', ''),
                        'doi': entity.get('doi', ''),
                        'journal': entity.get('journal', ''),
                        'year': entity.get('year', ''),
                        'page': entity.get('page', ''),
                        'source': entity.get('source') or entity.get('title', ''),
                        'retrieval_stage': 'focused_fulltext'
                    }
                    doc = Document(
                        page_content=str(content), 
                        metadata=metadata
                    )
                    documents.append(doc)
            
            return documents

        except Exception as e:
            print(f"Error in routing retrieval Stage 2: {e}")
            # Fallback strategy
            return abstract_docs

def get_routing_retriever(connection_args, abstract_retriever, fulltext_col_name, embedding_function, fulltext_top_k=6):
    """
    Factory function to create the RoutingRetriever
    """
    return RoutingRetriever(
        abstract_retriever=abstract_retriever,
        connection_args=connection_args,
        fulltext_col_name=fulltext_col_name,
        embedding_function=embedding_function,
        fulltext_top_k=fulltext_top_k
    )

def get_prompt(prompt='', return_format=True):
    if prompt == '':
        Prompt = """Use the following pieces of context to answer the question at the end.
                    You should answer the question in detail as far as possible.
                    If you cannot find anwser in the context, just say that you don't know, don't try to make up an answer.

                    {context}

                    Question: {question}

                    Helpful Answer:
                """
    if return_format:
        Prompt = PromptTemplate.from_template(prompt)
    return Prompt

# def get_llm_LLaMA(model_name, model_path, n_batch=2048, n_ctx=8192, verbose=False, 
#                   streaming=True, max_tokens=8192, temperature=0.8):
#     if model_name == 'LLaMA3-8B':
#         tokenizer = AutoTokenizer.from_pretrained('/data-10gb/data/llm/llama3/Meta-Llama-3-8B-Instruct-hf')
#         terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#         model_kwargs = {'do_sample':True, 'eos_token_id':terminators, 'max_new_tokens':8192, 'pad_token_id':128001}
#         llm = LlamaCpp(model_path=model_path, 
#                        n_gpu_layers=-1,
#                        n_ctx=8192, 
#                        n_batch=n_batch, 
#                        f16_kv=True,
#                        verbose=verbose,
#                        streaming=streaming, 
#                        temperature=temperature,
#                        model_kwargs=model_kwargs)
#         llm.client.verbose=False
#     elif model_name == 'LLaMA2-7B':
#         llm = LlamaCpp(model_path=model_path,
#                        n_gpu_layers=-1,
#                        n_ctx=n_ctx,
#                        n_batch=n_batch,
#                        f16_kv=True,
#                        verbose=verbose,
#                        streaming=streaming,
#                        temperature=temperature,
#                        max_tokens=max_tokens)
#     llm.client.verbose=False
#     return llm

def get_llm_ollama(model_name, num_predict, num_ctx=8192, keep_alive=600, temperature=0.1, base_url='http://10.15.102.186:9000'):
    if model_name == 'LLaMA3-8B':
        model = 'llama3:8b-instruct-q8_0'
    elif model_name == 'LLaMA2-7B':
        model = 'llama2:7b-chat-q8_0'
    elif model_name == 'LLaMA3.1-8B':
        model = 'llama3.1:8b-instruct-q8_0'
    elif model_name == 'Qwen2-7B':
        model = 'qwen2:7b-instruct-q8_0'
    elif model_name == 'Qwen2.5-7B':
        model = 'qwen2.5:7b-instruct-q8_0'
    elif model_name == 'Qwen2.5-14B':
        model = 'qwen2.5:14b-instruct-q8_0'
    elif model_name == 'Qwen2.5-72B':
        model = 'qwen2.5:72b-instruct-q8'
    elif model_name == 'Qwen2.5-32B':
        model = 'qwen3:30b-a3b-instruct-2507-q8_0'
    elif model_name == 'Qwen3-30B-Instruct':
        model = 'qwen3:30b-a3b-instruct-2507-q8_0'
    elif model_name == 'Qwen3-30B-thinking':
        model = 'qwen3:30b-a3b-thinking-2507-q8_0'
    llm = ChatOllama(model=model, num_ctx=num_ctx, keep_alive=keep_alive, num_predict=num_predict, 
                     temperature=temperature, base_url=base_url, num_thread=2)
    return llm

def get_contextualize_question(llm, history_prompt_template, input_: dict):
    history_context = None
    history_chain = history_prompt_template | llm | StrOutputParser()
    if input_.get('chat_history'):
        history_context = history_chain
    else:
        history_context = input_['question']
    return history_context

def retrieve_generate(question, llm, prompt, retriever, history=None, return_source=True, return_chain=False, use_query_rewrite=False):
    """
    question
        │
        ▼
    Retriever (检索 Milvus 文献 chunks)
        │
        ▼
    文献 chunks + 原始 question 打包成一个 dict
        │
        ▼
    RunnablePassthrough.assign(context=format_docs())
    （把文献列表转换成一大段文本）
        │
        ▼
    prompt （把 context 和 question 放进 prompt 模板）
        │
        ▼
    LLM (Ollama)
        │
        ▼
    StrOutputParser （把 LLM 输出变成字符串）
        │
        ▼
    最终返回 answer

    """
    
    # use_query_rewrite logic
    original_question = question
    rewritten_question = None  # Track if query was rewritten
    if use_query_rewrite:
        try:
            rewritten = rewrite_query(llm, original_question, history)
            question = rewritten
            rewritten_question = rewritten
        except Exception as e:
            print("Query rewrite failed, using original question.", e)
            question = original_question
            rewritten_question = None

    if return_source:
        rag_source = (RunnablePassthrough.assign(
            context=(lambda x: utils.format_docs(x['context'])))
            | prompt
            | llm
            | StrOutputParser()
        )

        if history:
            # Wrap history string in RunnableLambda to make it compatible with RunnableParallel
            history_runnable = RunnableLambda(lambda x: history)
            rag_chain = RunnableParallel(
                {'context':retriever, 'history':history_runnable, 'question':RunnablePassthrough()}).assign(
                    answer=rag_source)
        else:
            rag_chain = RunnableParallel(
                {'context':retriever, 'question':RunnablePassthrough()}).assign(
                    answer=rag_source)

    else:
        if history:
            # Wrap history string in RunnableLambda to make it compatible with RunnableParallel
            history_runnable = RunnableLambda(lambda x: history)
            rag_chain = ({'context':retriever, 'history':history_runnable, 'question':RunnablePassthrough()}
                         | prompt | llm)
        else:
            rag_chain = ({'context':retriever, 'question':RunnablePassthrough()}
                         | prompt | llm)

    if return_chain:
        return rag_chain
    else:
        answer = rag_chain.invoke(question)

        # Strip thinking tags from thinking models (e.g., qwen3-30B-thinking)
        # This removes CoT content before </think> tag
        if isinstance(answer, dict):
            # When return_source=True, answer is a dict with 'answer' key
            if 'answer' in answer:
                answer['answer'] = utils.strip_thinking_tags(answer['answer'])
        elif hasattr(answer, 'content'):
            # When return_source=False, answer might be an AIMessage
            answer.content = utils.strip_thinking_tags(answer.content)
        elif isinstance(answer, str):
            # When return_source=False, answer might be a string
            answer = utils.strip_thinking_tags(answer)

        # Add rewritten query info to the answer if applicable
        if rewritten_question:
            # Only attach when the answer is a dict (return_source=True path)
            if isinstance(answer, dict):
                answer['rewritten_query'] = rewritten_question
        return answer
