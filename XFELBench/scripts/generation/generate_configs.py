#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config Generator for XFELBench
Generates multiple configuration files for systematic RAG evaluation
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


# Base configuration template
BASE_CONFIG = {
    "model": {
        "llm_name": "Qwen3-30B-Instruct",
        "embedding_model": "BGE-M3",
        "temperature": 0.1,
        "num_predict": 2048,
        "num_ctx": 8192
    },
    "database": {
        "milvus": {
            "host": "10.19.48.181",
            "port": 19530,
            "username": "cs286_2025_group8",
            "password": "Group8",
            "db_name": "cs286_2025_group8"
        }
    },
    "collection": {
        "name": "xfel_bibs_collection_with_abstract"
    },
    "year_filter": {
        "enabled": True,
        "start_year": 2000,
        "end_year": 2025
    },
    "retrieval": {
        "top_k": 10,
        "search_params": {
            "ef": 20
        }
    },
    "prompt": {
        "template_file": "prompts/naive.pt"
    },
    "evaluation": {
        "batch_size": 10,
        "save_sources": True,
        "save_rewritten_queries": False
    }
}


# Configuration presets for different experimental settings
EXPERIMENT_CONFIGS = {
    "baseline": {
        "name": "baseline",
        "description": "Baseline: Dense search + Reranking only",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {"enabled": False},
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        }
    },

    "no_rerank": {
        "name": "no_rerank",
        "description": "Simplest: Dense search only, no reranking",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {"enabled": False},
            "rerank": {"enabled": False},
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        }
    },

    "hybrid_search": {
        "name": "hybrid_search",
        "description": "Hybrid: Dense + Sparse search with reranking",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {
                "enabled": True,
                "dense_weight": 0.5,
                "sparse_weight": 0.5
            },
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        }
    },

    "hybrid_dense_heavy": {
        "name": "hybrid_dense_heavy",
        "description": "Hybrid search with dense weight = 0.7",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {
                "enabled": True,
                "dense_weight": 0.7,
                "sparse_weight": 0.3
            },
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        }
    },

    "hybrid_sparse_heavy": {
        "name": "hybrid_sparse_heavy",
        "description": "Hybrid search with sparse weight = 0.7",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {
                "enabled": True,
                "dense_weight": 0.3,
                "sparse_weight": 0.7
            },
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        }
    },

    "query_rewrite": {
        "name": "query_rewrite",
        "description": "Baseline + Query rewriting",
        "features": {
            "query_rewrite": {"enabled": True},
            "hybrid_search": {"enabled": False},
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        },
        "evaluation": {
            "save_rewritten_queries": True
        }
    },

    "routing": {
        "name": "routing",
        "description": "Baseline + Two-stage routing",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {"enabled": False},
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {
                "enabled": True,
                "fulltext_top_k": 6
            },
            "chat_history": {"enabled": False}
        }
    },

    "hybrid_rewrite": {
        "name": "hybrid_rewrite",
        "description": "Hybrid search + Query rewriting",
        "features": {
            "query_rewrite": {"enabled": True},
            "hybrid_search": {
                "enabled": True,
                "dense_weight": 0.5,
                "sparse_weight": 0.5
            },
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        },
        "evaluation": {
            "save_rewritten_queries": True
        }
    },

    "hybrid_routing": {
        "name": "hybrid_routing",
        "description": "Hybrid search + Two-stage routing",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {
                "enabled": True,
                "dense_weight": 0.5,
                "sparse_weight": 0.5
            },
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {
                "enabled": True,
                "fulltext_top_k": 6
            },
            "chat_history": {"enabled": False}
        }
    },

    "full_features": {
        "name": "full_features",
        "description": "All features: Query Rewrite + Hybrid + Reranking + Routing",
        "features": {
            "query_rewrite": {"enabled": True},
            "hybrid_search": {
                "enabled": True,
                "dense_weight": 0.5,
                "sparse_weight": 0.5
            },
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 6
            },
            "routing": {
                "enabled": True,
                "fulltext_top_k": 6
            },
            "chat_history": {"enabled": False}
        },
        "evaluation": {
            "save_rewritten_queries": True
        }
    },

    "rerank_top3": {
        "name": "rerank_top3",
        "description": "Baseline with reranker top_n=3",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {"enabled": False},
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 3
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        }
    },

    "rerank_top10": {
        "name": "rerank_top10",
        "description": "Baseline with reranker top_n=10",
        "features": {
            "query_rewrite": {"enabled": False},
            "hybrid_search": {"enabled": False},
            "rerank": {
                "enabled": True,
                "model": "BAAI/bge-reranker-v2-m3",
                "top_n": 10
            },
            "routing": {"enabled": False},
            "chat_history": {"enabled": False}
        }
    },
}


def generate_config(experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a complete configuration by merging base config with experiment-specific settings.

    Args:
        experiment_config: Experiment-specific configuration

    Returns:
        Complete configuration dictionary
    """
    import copy
    config = copy.deepcopy(BASE_CONFIG)

    # Add experiment metadata
    config["experiment"] = {
        "name": experiment_config["name"],
        "description": experiment_config["description"],
        "version": "1.0"
    }

    # Add features
    config["features"] = experiment_config["features"]

    # Override evaluation settings if specified
    if "evaluation" in experiment_config:
        config["evaluation"].update(experiment_config["evaluation"])

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"[INFO] Generated: {output_path}")


def generate_all_configs(output_dir: str = "configs/generated",
                         selected_configs: List[str] = None):
    """
    Generate all configuration files.

    Args:
        output_dir: Directory to save generated configs (relative to XFELBench root)
        selected_configs: List of config names to generate (None = all)
    """
    # Get XFELBench root (script is in scripts/generation/, go up 2 levels)
    xfelbench_root = Path(__file__).parent.parent.parent
    output_path = xfelbench_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter configs if specified
    if selected_configs:
        configs_to_generate = {k: v for k, v in EXPERIMENT_CONFIGS.items()
                              if k in selected_configs}
    else:
        configs_to_generate = EXPERIMENT_CONFIGS

    print(f"[INFO] Generating {len(configs_to_generate)} configuration files...")
    print(f"[INFO] Output directory: {output_path}")

    generated_files = []

    for exp_name, exp_config in configs_to_generate.items():
        config = generate_config(exp_config)
        output_file = output_path / f"{exp_name}.yaml"
        save_config(config, str(output_file))
        generated_files.append(str(output_file))

    print(f"\n[INFO] Successfully generated {len(generated_files)} configs:")
    for f in generated_files:
        print(f"  - {f}")

    # Generate a summary file
    summary = {
        "total_configs": len(generated_files),
        "configs": {name: cfg["description"] for name, cfg in configs_to_generate.items()}
    }

    summary_file = output_path / "CONFIG_SUMMARY.yaml"
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n[INFO] Summary saved to: {summary_file}")

    return generated_files


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate XFELBench configuration files')
    parser.add_argument('--output-dir', type=str, default='configs/generated',
                       help='Output directory for generated configs')
    parser.add_argument('--configs', type=str, nargs='+',
                       help='Specific configs to generate (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available config templates')

    args = parser.parse_args()

    if args.list:
        print("\n[INFO] Available configuration templates:\n")
        for name, cfg in EXPERIMENT_CONFIGS.items():
            print(f"  {name:25s} - {cfg['description']}")
        print(f"\n[INFO] Total: {len(EXPERIMENT_CONFIGS)} templates")
    else:
        generate_all_configs(args.output_dir, args.configs)
