import os

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.config import cli


def setup_vllm_env(config: InferenceConfig):
    """Set vLLM environment variables based on config. Must be called before importing vLLM."""

    # spawn is more robust in vLLM nightlies and Qwen3-VL (fork can deadlock with multithreaded processes)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Large MoE models (e.g. GLM-4.7-Flash, 62.5GB on TP=4) take longer than
    # vLLM's default 600s engine-core startup timeout to load weights + capture
    # CUDA graphs. Exceeding it kills the engine core, so the server never
    # serves /health and every rollout comes back empty. Bump to 30 min.
    os.environ.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1800")

    if config.enable_lora:
        os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"


def main():
    config = cli(InferenceConfig)
    setup_vllm_env(config)

    # We import here to be able to set environment variables before importing vLLM
    from prime_rl.inference.vllm.server import server  # pyright: ignore

    server(config, vllm_extra=config.vllm_extra)


if __name__ == "__main__":
    main()
