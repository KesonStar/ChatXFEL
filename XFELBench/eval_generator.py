#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XFELBench Evaluation Generator
Generates answers for question sets using configured RAG pipeline
"""

import os
import sys
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

# Add parent directory to path to import ChatXFEL modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rag
import utils
from query_rewriter import rewrite_query
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class RAGEvaluationGenerator:
    """
    Generator for RAG evaluation experiments.
    Loads configuration, initializes RAG pipeline, and generates answers for question sets.
    """

    def __init__(self, config_path: str):
        """
        Initialize the evaluation generator with a configuration file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.output_dir = None

        # Initialize components
        self.embedding = None
        self.llm = None
        self.retriever = None
        self.prompt = None

        print(f"[INFO] Loaded configuration: {self.config['experiment']['name']}")
        print(f"[INFO] Description: {self.config['experiment']['description']}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_output_directory(self) -> str:
        """Create output directory for this evaluation run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config['experiment']['name']
        output_dir = Path(__file__).parent / 'outputs' / f"{timestamp}_{exp_name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration to output directory
        config_output_path = output_dir / 'config.yaml'
        with open(config_output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

        self.output_dir = output_dir
        print(f"[INFO] Output directory: {output_dir}")
        return str(output_dir)

    def initialize_components(self):
        """Initialize all RAG components based on configuration"""
        print("[INFO] Initializing RAG components...")

        # 1. Initialize embedding model
        print(f"[INFO] Loading embedding model: {self.config['model']['embedding_model']}")
        self.embedding = rag.get_embedding_bge()

        # 2. Initialize LLM
        llm_name = self.config['model']['llm_name']
        temperature = self.config['model']['temperature']
        num_predict = self.config['model']['num_predict']
        num_ctx = self.config['model']['num_ctx']

        print(f"[INFO] Loading LLM: {llm_name} (temperature={temperature})")
        self.llm = rag.get_llm_ollama(
            model_name=llm_name,
            num_predict=num_predict,
            num_ctx=num_ctx,
            keep_alive=-1,
            temperature=temperature,
            base_url='http://10.15.102.186:9000'
        )

        # 3. Initialize prompt template
        prompt_file = self.config['prompt']['template_file']
        if not os.path.isabs(prompt_file):
            # Relative path from project root
            prompt_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                prompt_file
            )

        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        self.prompt = rag.get_prompt(prompt_template)
        print(f"[INFO] Loaded prompt template: {prompt_file}")

        # 4. Get Milvus connection args
        milvus_config = self.config['database']['milvus']
        connection_args = utils.get_milvus_connection(
            host=milvus_config['host'],
            port=milvus_config['port'],
            username=milvus_config.get('username'),
            password=milvus_config.get('password'),
            db_name=milvus_config.get('db_name')
        )

        # 5. Initialize retriever based on configuration
        self._initialize_retriever(connection_args)

        print("[INFO] All components initialized successfully!")

    def _initialize_retriever(self, connection_args: Dict[str, Any]):
        """
        Initialize retriever with configured features.
        This method replicates the logic from chatxfel_app.py with configurable switches.
        """
        col_name = self.config['collection']['name']
        features = self.config['features']

        # Prepare year filter if enabled
        filters = {}
        if self.config['year_filter']['enabled']:
            year_start = self.config['year_filter']['start_year']
            year_end = self.config['year_filter']['end_year']
            filters['expr'] = f'{year_start} <= year <= {year_end}'
            print(f"[INFO] Year filter enabled: {year_start} - {year_end}")

        # Check if hybrid search is enabled
        use_hybrid = features['hybrid_search']['enabled']
        use_rerank = features['rerank']['enabled']
        use_routing = features['routing']['enabled']

        if use_hybrid:
            print("[INFO] Initializing HYBRID retriever (dense + sparse)...")
            # Load BGE-M3 for hybrid search
            embedding_m3 = rag.get_embedding_bge_m3()

            dense_weight = features['hybrid_search']['dense_weight']
            sparse_weight = features['hybrid_search']['sparse_weight']
            top_n = features['rerank']['top_n']

            filter_expr = filters.get('expr', None)

            self.retriever = rag.get_hybrid_retriever(
                connection_args=connection_args,
                col_name=col_name,
                embedding=embedding_m3,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                use_rerank=use_rerank,
                top_n=top_n,
                filters=filter_expr
            )
            print(f"[INFO] Hybrid retriever created (dense_weight={dense_weight}, sparse_weight={sparse_weight})")

            # Store embedding_m3 for potential routing use
            self._embedding_for_routing = embedding_m3
        else:
            print("[INFO] Initializing DENSE-ONLY retriever...")
            # Use standard dense retriever
            retriever_obj = rag.get_retriever(
                connection_args=connection_args,
                col_name=col_name,
                embedding=self.embedding,
                vector_field='dense_vector',
                use_rerank=False,
                return_as_retreiever=False
            )

            # Apply reranking if enabled
            if use_rerank:
                top_n = features['rerank']['top_n']
                rerank_model_name = features['rerank']['model']

                print(f"[INFO] Adding reranker: {rerank_model_name} (top_n={top_n})")
                rerank_model = HuggingFaceCrossEncoder(model_name=rerank_model_name)
                compressor = CrossEncoderReranker(model=rerank_model, top_n=top_n)

                # Wrap retriever with compression
                search_kwargs = {'k': self.config['retrieval']['top_k']}
                if filters:
                    search_kwargs = {**search_kwargs, **filters}

                base_retriever = retriever_obj.as_retriever(search_kwargs=search_kwargs)
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
            else:
                search_kwargs = {'k': self.config['retrieval']['top_k']}
                if filters:
                    search_kwargs = {**search_kwargs, **filters}
                self.retriever = retriever_obj.as_retriever(search_kwargs=search_kwargs)

            # Store embedding for potential routing use
            self._embedding_for_routing = self.embedding

        # Apply routing if enabled
        if use_routing:
            print("[INFO] Applying ROUTING (two-stage retrieval)...")
            fulltext_top_k = features['routing']['fulltext_top_k']

            self.retriever = rag.get_routing_retriever(
                connection_args=connection_args,
                abstract_retriever=self.retriever,
                fulltext_col_name=col_name,
                embedding_function=self._embedding_for_routing,
                fulltext_top_k=fulltext_top_k
            )
            print(f"[INFO] Routing retriever created (fulltext_top_k={fulltext_top_k})")

    def load_questions(self, question_file: str) -> List[Dict[str, Any]]:
        """
        Load questions from JSON file.

        Args:
            question_file: Path to question JSON file

        Returns:
            List of question dictionaries
        """
        with open(question_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        questions = data.get('questions', [])
        print(f"[INFO] Loaded {len(questions)} questions from {question_file}")
        return questions

    def generate_answer(self, question: str, question_id: str = None) -> Dict[str, Any]:
        """
        Generate answer for a single question using the RAG pipeline.

        Args:
            question: Question text
            question_id: Optional question ID for logging

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        use_query_rewrite = self.config['features']['query_rewrite']['enabled']

        # Build history (empty for now, can be extended for chat history feature)
        history = None
        if self.config['features']['chat_history']['enabled']:
            history = "No previous conversation."  # Placeholder

        # Generate answer using RAG
        start_time = time.time()
        output = rag.retrieve_generate(
            question=question,
            llm=self.llm,
            prompt=self.prompt,
            retriever=self.retriever,
            history=history,
            return_source=True,
            return_chain=False,
            use_query_rewrite=use_query_rewrite
        )
        end_time = time.time()

        # Extract answer and context
        answer = output.get('answer', '')
        context_docs = output.get('context', [])
        rewritten_query = output.get('rewritten_query', None)

        # Format sources
        sources = []
        for doc in context_docs:
            source_info = {
                'title': doc.metadata.get('title', ''),
                'doi': doc.metadata.get('doi', ''),
                'journal': doc.metadata.get('journal', ''),
                'year': doc.metadata.get('year', ''),
                'page': doc.metadata.get('page', ''),
                'content': doc.page_content
            }
            sources.append(source_info)

        result = {
            'question_id': question_id,
            'question': question,
            'answer': answer,
            'sources': sources if self.config['evaluation']['save_sources'] else [],
            'rewritten_query': rewritten_query if self.config['evaluation']['save_rewritten_queries'] else None,
            'generation_time': end_time - start_time,
            'timestamp': datetime.now().isoformat()
        }

        return result

    def run_evaluation(self, question_file: str):
        """
        Run evaluation on a question set.

        Args:
            question_file: Path to question JSON file
        """
        # Setup output directory
        self._setup_output_directory()

        # Load questions
        questions = self.load_questions(question_file)

        # Results file
        results_file = self.output_dir / 'results.jsonl'

        print(f"[INFO] Starting evaluation on {len(questions)} questions...")
        print(f"[INFO] Results will be saved to: {results_file}")

        # Process questions with progress bar
        batch_size = self.config['evaluation']['batch_size']

        with open(results_file, 'w', encoding='utf-8') as f:
            for i, q_data in enumerate(tqdm(questions, desc="Generating answers")):
                question_id = q_data.get('id', f'q_{i}')
                question_text = q_data['question']

                try:
                    # Generate answer
                    result = self.generate_answer(question_text, question_id)

                    # Add question metadata
                    result['metadata'] = {
                        'category': q_data.get('category'),
                        'difficulty': q_data.get('difficulty'),
                        'expected_topics': q_data.get('expected_topics', [])
                    }

                    # Write result to JSONL file (one line per question)
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is written immediately

                    # Log progress every batch_size questions
                    if (i + 1) % batch_size == 0:
                        print(f"[INFO] Processed {i + 1}/{len(questions)} questions")

                except Exception as e:
                    print(f"[ERROR] Failed to process question {question_id}: {e}")
                    # Write error result
                    error_result = {
                        'question_id': question_id,
                        'question': question_text,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                    f.flush()

        # Generate summary
        self._generate_summary(results_file, len(questions))

        print(f"[INFO] Evaluation completed!")
        print(f"[INFO] Results saved to: {self.output_dir}")

    def _generate_summary(self, results_file: Path, total_questions: int):
        """Generate summary statistics for the evaluation run"""
        summary = {
            'experiment_name': self.config['experiment']['name'],
            'total_questions': total_questions,
            'completed_questions': 0,
            'failed_questions': 0,
            'average_generation_time': 0.0,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        # Read results and compute statistics
        generation_times = []
        completed = 0
        failed = 0

        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                if 'error' in result:
                    failed += 1
                else:
                    completed += 1
                    if 'generation_time' in result:
                        generation_times.append(result['generation_time'])

        summary['completed_questions'] = completed
        summary['failed_questions'] = failed
        if generation_times:
            summary['average_generation_time'] = sum(generation_times) / len(generation_times)

        # Save summary
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Summary: {completed}/{total_questions} completed, {failed} failed")
        print(f"[INFO] Average generation time: {summary['average_generation_time']:.2f}s")


def main():
    """Main entry point for evaluation generator"""
    parser = argparse.ArgumentParser(description='XFELBench Evaluation Generator')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--questions', type=str, required=True,
                       help='Path to question set JSON file')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.config):
        print(f"[ERROR] Configuration file not found: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.questions):
        print(f"[ERROR] Question file not found: {args.questions}")
        sys.exit(1)

    # Initialize generator
    generator = RAGEvaluationGenerator(args.config)

    # Initialize components
    generator.initialize_components()

    # Run evaluation
    generator.run_evaluation(args.questions)


if __name__ == '__main__':
    main()
