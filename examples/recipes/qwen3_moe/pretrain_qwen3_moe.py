#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3 MoE Pretraining Script with YAML and CLI Configuration Overrides.

This script provides a flexible way to pretrain Qwen3 MoE models (30B-A3B or 235B-A22B)
using Megatron-Bridge with support for both YAML configuration files and command-line
overrides using Hydra-style syntax.

Examples:
    Basic usage with Qwen3-30B-A3B (default):
        $ torchrun --nproc_per_node=8 pretrain_qwen3_moe.py

    Using Qwen3-235B-A22B:
        $ torchrun --nproc_per_node=8 pretrain_qwen3_moe.py --model 235b

    Using CLI overrides:
        $ torchrun --nproc_per_node=8 pretrain_qwen3_moe.py \\
            model.tensor_model_parallel_size=4 \\
            model.expert_model_parallel_size=8 \\
            train.train_iters=100000

    With mock data for testing:
        $ torchrun --nproc_per_node=8 pretrain_qwen3_moe.py dataset.mock=true
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Tuple

import debugpy
import torch
from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen.qwen3_moe import (
    qwen3_30b_a3b_pretrain_config,
    qwen3_235b_a22b_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)


logger: logging.Logger = logging.getLogger(__name__)


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments, separating known script args from OmegaConf overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Qwen3 MoE model using Megatron-Bridge with YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="30b",
        choices=["30b", "235b"],
        help="Model variant to use: '30b' for Qwen3-30B-A3B, '235b' for Qwen3-235B-A22B (default: 30b)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the YAML OmegaConf override file (optional)",
    )
    parser.add_argument("--per-split-data-args-path", type=str, help="Path to the per split data args file.")
    parser.add_argument("--tokenizer-model", type=str, help="Path or HF model ID for the tokenizer.")

    # Parse known args for the script, remaining will be treated as overrides
    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Entry point for the Qwen3 MoE pretraining script.
    """
    if os.environ.get("DEBUG", "") == "1":
        rank = int(os.environ.get("RANK", "0"))
        port = 5678 + rank
        debugpy.listen(("0.0.0.0", port))
        if rank == 0:
            logger.info(f"⏳ Rank 0 waiting for debugger on port {port}...")
            debugpy.wait_for_client()
            logger.info("🔗 Rank 0 debugger attached!")
        else:
            logger.info(f"Rank {rank} debugpy listening on port {port} (attach anytime)")

    args, cli_overrides = parse_cli_args()

    # Select the appropriate config based on model variant
    if args.model == "235b":
        cfg: ConfigContainer = qwen3_235b_a22b_pretrain_config(
            per_split_data_args_path=args.per_split_data_args_path,
        )
        logger.info("Using Qwen3-235B-A22B configuration")
    else:
        cfg: ConfigContainer = qwen3_30b_a3b_pretrain_config(
            per_split_data_args_path=args.per_split_data_args_path,
        )
        logger.info("Using Qwen3-30B-A3B configuration")

    # Convert the initial Python dataclass to an OmegaConf DictConfig for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Load and merge YAML overrides if a config file is provided
    if args.config_file:
        logger.debug(f"Loading YAML overrides from: {args.config_file}")
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides_omega = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides_omega)
        logger.debug("YAML overrides merged successfully.")

    # Apply command-line overrides using Hydra-style parsing
    if cli_overrides:
        logger.debug(f"Applying Hydra-style command-line overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("Hydra-style command-line overrides applied successfully.")

    # Apply the final merged OmegaConf configuration back to the original ConfigContainer
    logger.debug("Applying final merged configuration back to Python ConfigContainer...")
    final_overrides_as_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    # Apply overrides while preserving excluded fields
    apply_overrides(cfg, final_overrides_as_dict, excluded_fields)

    # PP_LAYOUT env var bypasses Hydra (special chars like |, *, () break Hydra grammar)
    if pp_layout := os.environ.get("PP_LAYOUT"):
        cfg.model.pipeline_model_parallel_layout = pp_layout
        logger.info(f"Using PP_LAYOUT from env: {pp_layout}")

    # Start training
    logger.debug("Starting pretraining...")
    if os.environ.get("DEBUG", "") == "1":
        try:
            pretrain(config=cfg, forward_step_func=forward_step)
        except Exception as e:
            rank = int(os.environ.get("RANK", "0"))
            logger.error(f"Rank {rank} caught exception: {e}")
            debugpy.breakpoint()  # debugger will pause here — inspect `e`
            # Keep process alive so other ranks don't get killed by NCCL timeout
            logger.info(f"Rank {rank} holding process alive for debugging. Ctrl+C or kill to exit.")
            signal.pause()
    else:
        pretrain(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
