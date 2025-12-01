#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Implementation for ChatXFEL utilities
# Date: 2025

"""
Utility functions for ChatXFEL project
Provides MongoDB and Milvus connection utilities, logging, and document formatting
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pymongo import MongoClient
from langchain_classic.schema.document import Document


def get_mongodb_client(db_name: str, host: str = 'localhost', port: int = 27017, 
                       username: Optional[str] = None, password: Optional[str] = None) -> MongoClient:
    """
    Get MongoDB client connection
    
    Args:
        db_name: Database name
        host: MongoDB host address (default: localhost)
        port: MongoDB port (default: 27017)
        username: MongoDB username (optional)
        password: MongoDB password (optional)
    
    Returns:
        MongoClient: MongoDB client instance
    
    Example:
        client = get_mongodb_client(db_name='test_doi')
        db = client.get_database('test_doi')
        col = db.get_collection('bibs')
    """
    if username and password:
        # Connection with authentication
        connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
        client = MongoClient(connection_string)
    else:
        # Connection without authentication
        client = MongoClient(host=host, port=port)
    
    # Test connection
    client.server_info()
    print(f"Successfully connected to MongoDB at {host}:{port}")
    return client


def get_milvus_connection(host: str = '10.19.48.181', port: int = 19530,
                         username: Optional[str] = None, 
                         password: Optional[str] = None,
                         db_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get Milvus connection arguments
    
    Args:
        host: Milvus host address (default: 10.19.48.181)
        port: Milvus port (default: 19530)
        username: Milvus username (optional)
        password: Milvus password (optional)
        db_name: Milvus database name (optional but recommended)
    
    Returns:
        Dict containing connection arguments for Milvus
    
    Example:
        connection_args = get_milvus_connection(
            host='10.19.48.181',
            port=19530,
            username='cs286_2025_group8',
            password='Group8',
            db_name='cs286_2025_group8'
        )
        retriever = Milvus(
            embedding_function=embedding,
            connection_args=connection_args,
            collection_name='chatxfel'
        )
    """
    connection_args = {
        'host': host,
        'port': str(port)  # 转换为字符串，与你的代码一致
    }
    
    if username and password:
        connection_args['user'] = username
        connection_args['password'] = password
    
    if db_name:
        connection_args['db_name'] = db_name
    
    # 内网一般不启用TLS
    connection_args['secure'] = False
    
    return connection_args



def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a single string for context
    
    Args:
        docs: List of Document objects from retriever
    
    Returns:
        Formatted string containing all document contents
    
    Example:
        formatted = format_docs(retrieved_docs)
        # Output: "Document 1 content\n\nDocument 2 content\n\n..."
    """
    if not docs:
        return ""
    
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        
        # Optionally add metadata information
        metadata = doc.metadata
        if metadata:
            title = metadata.get('title', 'Unknown')
            page = metadata.get('page', 'N/A')
            formatted_parts.append(
                f"[Document {i} - {title}, Page {page}]\n{content}"
            )
        else:
            formatted_parts.append(f"[Document {i}]\n{content}")
    
    return "\n\n".join(formatted_parts)


def log_rag(log_data: Dict[str, Any], use_mongo: bool = True,
           db_name: str = 'chatxfel_logs', collection_name: str = 'rag_logs',
           log_file: Optional[str] = None) -> bool:
    """
    Log RAG query, answer, and feedback to MongoDB and/or file
    
    Args:
        log_data: Dictionary containing log information
                 Expected keys: IP, Time, Model, Question, Answer, Source, Feedback
        use_mongo: Whether to log to MongoDB (default: True)
        db_name: MongoDB database name (default: 'chatxfel_logs')
        collection_name: MongoDB collection name (default: 'rag_logs')
        log_file: Optional file path to also save logs
    
    Returns:
        bool: True if logging successful, False otherwise
    
    Example:
        logs = {
            'IP': '192.168.1.1',
            'Time': '2025-01-01 12:00:00',
            'Model': 'Qwen2.5-32B',
            'Question': 'What is XFEL?',
            'Answer': 'XFEL stands for...',
            'Source': 'Reference documents...',
            'Feedback': '5'
        }
        log_rag(logs, use_mongo=True)
    """
    # Add timestamp if not present
    if 'Timestamp' not in log_data:
        log_data['Timestamp'] = datetime.now()
    
    # Log to MongoDB
    if use_mongo:
        client = get_mongodb_client(db_name=db_name)
        db = client.get_database(db_name)
        collection = db.get_collection(collection_name)
        
        result = collection.insert_one(log_data)
        client.close()
        
        if result.acknowledged:
            print(f"Successfully logged to MongoDB: {result.inserted_id}")
        else:
            print("Failed to log to MongoDB")
            return False
    
    # Log to file if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {log_data.get('Timestamp', 'N/A')}\n")
            f.write(f"IP: {log_data.get('IP', 'N/A')}\n")
            f.write(f"Model: {log_data.get('Model', 'N/A')}\n")
            f.write(f"Question: {log_data.get('Question', 'N/A')}\n")
            f.write(f"Answer: {log_data.get('Answer', 'N/A')}\n")
            f.write(f"Source: {log_data.get('Source', 'N/A')}\n")
            f.write(f"Feedback: {log_data.get('Feedback', 'N/A')}\n")
            f.write(f"{'='*80}\n")
            
        print(f"Successfully logged to file: {log_file}")
    
    return True


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def get_collection_stats(connection_args: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
    """
    Get statistics about a Milvus collection
    
    Args:
        connection_args: Milvus connection arguments
        collection_name: Name of the collection
    
    Returns:
        Dictionary containing collection statistics
    """
    from pymilvus import connections, Collection
    
    # Connect to Milvus
    connections.connect(**connection_args)
    
    # Get collection
    collection = Collection(name=collection_name)
    collection.load()
    
    # Get stats
    stats = {
        'name': collection_name,
        'num_entities': collection.num_entities,
        'schema': str(collection.schema),
        'description': collection.description
    }
    
    connections.disconnect("default")
    
    return stats


def validate_doi(doi: str) -> bool:
    """
    Validate DOI format
    
    Args:
        doi: DOI string to validate
    
    Returns:
        bool: True if valid DOI format, False otherwise
    """
    import re
    
    if not doi:
        return False
    
    # Basic DOI pattern: 10.xxxx/xxxxx
    doi_pattern = r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$'
    
    return bool(re.match(doi_pattern, doi))


def merge_metadata(doc_metadata: Dict[str, Any], additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge document metadata with additional metadata
    
    Args:
        doc_metadata: Original document metadata
        additional_metadata: Additional metadata to merge
    
    Returns:
        Merged metadata dictionary
    """
    merged = doc_metadata.copy()
    merged.update(additional_metadata)
    return merged


# Configuration defaults
DEFAULT_MONGODB_CONFIG = {
    'host': 'localhost',
    'port': 27017,
    'db_name': 'test_doi'
}

DEFAULT_MILVUS_CONFIG = {
    'host': '10.15.85.78',
    'port': 19530
}

DEFAULT_LOG_CONFIG = {
    'db_name': 'chatxfel_logs',
    'collection_name': 'rag_logs',
    'log_dir': './logs'
}


if __name__ == "__main__":
    # Test functions
    print("Testing MongoDB connection...")
    client = get_mongodb_client(db_name='test_doi')
    print(f"MongoDB client created: {client}")
    client.close()
    
    print("\nTesting Milvus connection args...")
    connection_args = get_milvus_connection()
    print(f"Milvus connection args: {connection_args}")
    
    print("\nTesting format_docs...")
    test_docs = [
        Document(page_content="This is test document 1", metadata={'title': 'Test1', 'page': 1}),
        Document(page_content="This is test document 2", metadata={'title': 'Test2', 'page': 2})
    ]
    formatted = format_docs(test_docs)
    print(f"Formatted docs:\n{formatted}")
    
    print("\nTesting log_rag...")
    test_log = {
        'IP': '127.0.0.1',
        'Time': '2025-01-01 12:00:00',
        'Model': 'Test Model',
        'Question': 'Test question?',
        'Answer': 'Test answer',
        'Source': 'Test source',
        'Feedback': '5'
    }
    log_rag(test_log, use_mongo=False, log_file='./test_log.txt')
    
    print("\nAll tests completed!")