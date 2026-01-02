#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for XFELBench evaluation generator
Tests the generator with a minimal configuration on a few questions
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_generator import RAGEvaluationGenerator


def test_config_loading():
    """Test loading configuration files"""
    print("=" * 60)
    print("TEST 1: Configuration Loading")
    print("=" * 60)

    config_path = "configs/experiments/baseline.yaml"

    try:
        generator = RAGEvaluationGenerator(config_path)
        print("✓ Configuration loaded successfully")
        print(f"  Experiment name: {generator.config['experiment']['name']}")
        print(f"  LLM: {generator.config['model']['llm_name']}")
        print(f"  Hybrid search: {generator.config['features']['hybrid_search']['enabled']}")
        print(f"  Query rewrite: {generator.config['features']['query_rewrite']['enabled']}")
        print(f"  Rerank: {generator.config['features']['rerank']['enabled']}")
        print(f"  Routing: {generator.config['features']['routing']['enabled']}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_question_loading():
    """Test loading question sets"""
    print("\n" + "=" * 60)
    print("TEST 2: Question Set Loading")
    print("=" * 60)

    config_path = "configs/experiments/baseline.yaml"
    question_file = "problem_sets/xfel_qa_basic.json"

    try:
        generator = RAGEvaluationGenerator(config_path)
        questions = generator.load_questions(question_file)
        print(f"✓ Loaded {len(questions)} questions")

        if len(questions) > 0:
            print(f"\nSample question:")
            print(f"  ID: {questions[0].get('id')}")
            print(f"  Question: {questions[0]['question']}")
            print(f"  Category: {questions[0].get('category')}")
            print(f"  Difficulty: {questions[0].get('difficulty')}")

        return True
    except Exception as e:
        print(f"✗ Question loading failed: {e}")
        return False


def test_component_initialization():
    """Test initialization of RAG components"""
    print("\n" + "=" * 60)
    print("TEST 3: Component Initialization")
    print("=" * 60)
    print("NOTE: This test requires running services (Ollama, Milvus)")
    print("It may take a while to download models on first run...")

    config_path = "configs/experiments/baseline.yaml"

    try:
        generator = RAGEvaluationGenerator(config_path)
        print("\n[1/4] Initializing components...")
        generator.initialize_components()

        print("\n[2/4] Checking embedding model...")
        assert generator.embedding is not None
        print("✓ Embedding model initialized")

        print("\n[3/4] Checking LLM...")
        assert generator.llm is not None
        print("✓ LLM initialized")

        print("\n[4/4] Checking retriever...")
        assert generator.retriever is not None
        print("✓ Retriever initialized")

        return True
    except Exception as e:
        print(f"✗ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_question_generation():
    """Test generating an answer for a single question"""
    print("\n" + "=" * 60)
    print("TEST 4: Single Question Answer Generation")
    print("=" * 60)
    print("NOTE: This test requires all services to be running")

    config_path = "configs/experiments/baseline.yaml"

    try:
        # Initialize generator
        generator = RAGEvaluationGenerator(config_path)
        generator.initialize_components()

        # Test question
        test_question = "What is XFEL?"
        print(f"\nQuestion: {test_question}")

        # Generate answer
        print("\nGenerating answer...")
        result = generator.generate_answer(test_question, question_id="test_001")

        print("\n" + "-" * 60)
        print("RESULT:")
        print("-" * 60)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"\nGeneration time: {result['generation_time']:.2f}s")
        print(f"Number of sources: {len(result['sources'])}")

        if result.get('rewritten_query'):
            print(f"Rewritten query: {result['rewritten_query']}")

        if len(result['sources']) > 0:
            print(f"\nFirst source:")
            print(f"  Title: {result['sources'][0]['title']}")
            print(f"  DOI: {result['sources'][0]['doi']}")
            print(f"  Year: {result['sources'][0]['year']}")

        print("\n✓ Answer generated successfully")
        return True

    except Exception as e:
        print(f"✗ Answer generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("XFELBench Generator Test Suite")
    print("=" * 60)

    tests = [
        ("Configuration Loading", test_config_loading),
        ("Question Set Loading", test_question_loading),
    ]

    # Optional tests that require services
    run_service_tests = input("\nRun tests that require services (Ollama, Milvus)? [y/N]: ").lower() == 'y'

    if run_service_tests:
        tests.extend([
            ("Component Initialization", test_component_initialization),
            ("Single Question Generation", test_single_question_generation),
        ])

    # Run tests
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\nUnexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("=" * 60)


if __name__ == '__main__':
    # Change to XFELBench directory
    os.chdir(Path(__file__).parent)
    main()
