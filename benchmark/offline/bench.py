import argparse
import time
from random import randint, seed

import torch.profiler

from minisgl.core import SamplingParams
from minisgl.llm import LLM


def parse_args():
    parser = argparse.ArgumentParser(description="Offline throughput benchmark for mini-sglang")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model path or HuggingFace model name (default: Qwen/Qwen3-0.6B)",
    )

    parser.add_argument(
        "--num-seqs", type=int, default=256, help="Number of sequences to benchmark (default: 256)"
    )

    parser.add_argument(
        "--min-input-len",
        type=int,
        default=100,
        help="Minimum input length in tokens (default: 100)",
    )

    parser.add_argument(
        "--max-input-len",
        type=int,
        default=1024,
        help="Maximum input length in tokens (default: 1024)",
    )

    parser.add_argument(
        "--min-output-len",
        type=int,
        default=100,
        help="Minimum output length in tokens (default: 100)",
    )

    parser.add_argument(
        "--max-output-len",
        type=int,
        default=1024,
        help="Maximum output length in tokens (default: 1024)",
    )

    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature (default: 0.6)"
    )

    parser.add_argument(
        "--ignore-eos", action="store_true", default=True, help="Ignore EOS token (default: True)"
    )

    parser.add_argument(
        "--max-seq-len", type=int, default=4096, help="Maximum sequence length (default: 4096)"
    )

    parser.add_argument(
        "--max-extend-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to extend per iteration (default: 16384)",
    )

    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=256,
        help="Maximum batch size for CUDA graphs (default: 256)",
    )

    parser.add_argument(
        "--page-size", type=int, default=256, help="Page size for KV cache (default: 256)"
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    parser.add_argument(
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use random dummy weights instead of loading real weights",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler to capture execution trace",
    )

    parser.add_argument(
        "--profile-output",
        type=str,
        default="/tmp/trace.json",
        help="Profiler output file path (default: /tmp/trace.json)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the main benchmark run",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    seed(args.seed)

    print(f"Benchmark configuration:")
    print(f"  Model: {args.model}")
    print(f"  Sequences: {args.num_seqs}")
    print(f"  Input length: [{args.min_input_len}, {args.max_input_len}]")
    print(f"  Output length: [{args.min_output_len}, {args.max_output_len}]")
    print(f"  Temperature: {args.temperature}")
    print(f"  Ignore EOS: {args.ignore_eos}")
    print()

    llm = LLM(
        args.model,
        max_seq_len_override=args.max_seq_len,
        max_extend_tokens=args.max_extend_tokens,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        page_size=args.page_size,
        use_dummy_weight=args.use_dummy_weight,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]

    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=args.ignore_eos,
            max_tokens=randint(args.min_output_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))

    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_flops=True,
        ) as prof:
            llm.generate(prompt_token_ids, sampling_params)

        prof.export_chrome_trace(args.profile_output)
        print(f"Profile trace saved to {args.profile_output}")
        print()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    if not args.dry_run:
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params)
        t = time.time() - t

        total_tokens = sum(sp.max_tokens for sp in sampling_params)
        throughput = total_tokens / t

        print(f"Results:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Time: {t:.2f}s")
        print(f"  Throughput: {throughput:.2f} tok/s")


if __name__ == "__main__":
    main()
