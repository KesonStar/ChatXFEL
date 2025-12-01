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
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from transformers import AutoTokenizer
from langchain_community.chat_models import ChatOllama

sys.path.append('/home/zhangxf/workdir/LLM/llm-shine/ChatXFEL/src')
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

def load_pdf_directory(file_dir, recursive=True, multitread=True):
    kwargs = {'extract_images':True}
    loader = DirectoryLoader(file_dir, glob='**/*.pdf', loader_cls=PyPDFLoader, recursive=recursive,
                             loader_kwargs=kwargs, show_progress=True, use_multithreading=multitread)
    docs = loader.load()
    return docs

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
        model_kwargs = {'device':'mps'}
    if encode_kwargs is None:
        encode_kwargs = {'normalize_embeddings':True}
    model_name = 'BAAI/bge-m3'
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embedding

def get_embedding(model_name, n_gpu_layers=-1, n_ctx=4096):
    '''
    Supported models: llama, gpt
    '''
    embedding = None
    if 'llama' in model_name.lower():
        embedding = LlamaCppEmbeddings(
            model_path=model_name,
            n_gpu_layers = n_gpu_layers,
            n_ctx=n_ctx
        )
    elif 'gpt' in model_name.lower():
        print('Support for GPT models is TBD')
    else:
        print('Only gpt or llama are supported')

    return embedding

def restore_vector(docs, connection_args, col_name, embedding, desc=''):
    _ = Milvus(embedding_function=embedding,
                          connection_args=connection_args,
                          collection_name=col_name,
                          drop_old=True
                         ).from_documents(
                             docs,
                             embedding=embedding,
                             connection_args=connection_args,
                             collection_description=desc,
                             collection_name=col_name
                         )
    return _

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

def get_llm_LLaMA(model_name, model_path, n_batch=2048, n_ctx=8192, verbose=False, 
                  streaming=True, max_tokens=8192, temperature=0.8):
    if model_name == 'LLaMA3-8B':
        tokenizer = AutoTokenizer.from_pretrained('/data-10gb/data/llm/llama3/Meta-Llama-3-8B-Instruct-hf')
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        model_kwargs = {'do_sample':True, 'eos_token_id':terminators, 'max_new_tokens':8192, 'pad_token_id':128001}
        llm = LlamaCpp(model_path=model_path, 
                       n_gpu_layers=-1,
                       n_ctx=8192, 
                       n_batch=n_batch, 
                       f16_kv=True,
                       verbose=verbose,
                       streaming=streaming, 
                       temperature=temperature,
                       model_kwargs=model_kwargs)
        llm.client.verbose=False
    elif model_name == 'LLaMA2-7B':
        llm = LlamaCpp(model_path=model_path,
                       n_gpu_layers=-1,
                       n_ctx=n_ctx,
                       n_batch=n_batch,
                       f16_kv=True,
                       verbose=verbose,
                       streaming=streaming,
                       temperature=temperature,
                       max_tokens=max_tokens)
    llm.client.verbose=False
    return llm

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
    elif model_name == 'QwQ-32B':
        model = 'qwq:32b-preview-q8'
    elif model_name == 'Qwen3-30B':
        model = 'qwen3:30b-a3b-instruct-2507-q8_0'
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

def retrieve_generate(question, llm, prompt, retriever, history=None, return_source=True, return_chain=False): 
    if return_source:
        rag_source = (RunnablePassthrough.assign(
            context=(lambda x: utils.format_docs(x['context'])))
            | prompt
            | llm
            | StrOutputParser()
        )

        if history:
            rag_chain = RunnableParallel(
                {'context':retriever, 'history':history, 'question':RunnablePassthrough()}).assign(
                    answer=rag_source)
        else:
            rag_chain = RunnableParallel(
                {'context':retriever, 'question':RunnablePassthrough()}).assign(
                    answer=rag_source)

    else:
        if history:
            rag_chain = ({'context':retriever, 'history':history, 'question':RunnablePassthrough()}
                         | prompt | llm)
        else:
            rag_chain = ({'context':retriever, 'question':RunnablePassthrough()}
                         | prompt | llm)

    if return_chain:
        return rag_chain
    else:
        answer = rag_chain.invoke(question)
        return answer
