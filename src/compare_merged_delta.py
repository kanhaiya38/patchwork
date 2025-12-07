import torch
import time
import argparse
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# import sys
# sys.path.insert(0, '/storage/ice1/0/5/kmadaswar3/smr_project')
from src.validate import find_checkpoints
from src.constants import BASE_MODEL


def get_quantization_config(bits):
    """
    Get BitsAndBytesConfig for GPTQ-style quantization.

    Args:
        bits: 4 or 8 for quantization bits

    Returns:
        BitsAndBytesConfig for the specified bits
    """
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        raise ValueError(f"Unsupported bits: {bits}. Only 4 and 8 are supported.")


def load_model(model_path, quantization_bits=None):
    """
    Load a model with optional quantization.

    Args:
        model_path: Path to the model
        quantization_bits: None for no quantization, 4 or 8 for quantized loading

    Returns:
        Loaded model
    """
    if quantization_bits:
        print(f"  Loading with {quantization_bits}-bit quantization...")
        quantization_config = get_quantization_config(quantization_bits)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        print(f"  Loading without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    return model


def calculate_delta_nonzero_weights(base_model, merged_model):
    """
    Calculates the percentage of nonzero weights that differ between base and merged models.

    Args:
        base_model: The base model
        merged_model: The merged model (base + LoRA)

    Returns:
        float: The percentage of nonzero weight differences.
    """
    base_model_weights = base_model.state_dict()
    merged_model_weights = merged_model.state_dict()

    # Get all weight parameters (not just LoRA weights)
    weight_keys = [
        key for key in base_model_weights.keys()
        if key.endswith(".weight")
    ]

    total_nonzero = 0
    total_weights = 0

    print(f"Total weight parameters to compare: {len(weight_keys)}\n")

    for key in weight_keys:
        if key not in merged_model_weights:
            print(f"Warning: {key} not found in merged model, skipping...")
            continue

        start_time = time.time()
        print(f"Processing weights for: {key}")

        base_weights = base_model_weights[key].cpu().float()
        merged_weights = merged_model_weights[key].cpu().float()

        # Calculate weight difference
        weight_difference = torch.sub(base_weights, merged_weights)
        nonzero_count = torch.count_nonzero(weight_difference).item()
        total_nonzero += nonzero_count
        total_weights += base_weights.numel()

        # Print statistics for this layer
        if nonzero_count > 0:
            layer_percentage = (nonzero_count / base_weights.numel()) * 100
            print(f"  Nonzero differences: {nonzero_count}/{base_weights.numel()} ({layer_percentage:.4f}%)")

        end_time = time.time()
        print(f"  Time taken: {end_time - start_time:.2f}s\n")

    if total_weights == 0:
        return 0.0

    return (float(total_nonzero) / float(total_weights)) * 100


def run_comparison(base_model_name, merged_model_path, quantization_bits=None):
    """
    Run a single comparison between base and merged models with optional quantization.

    Args:
        base_model_name: Name or path of the base model
        merged_model_path: Path to the merged model
        quantization_bits: None for no quantization, 4 or 8 for quantized comparison

    Returns:
        float: Delta percentage
    """
    quantization_label = f"{quantization_bits}-bit" if quantization_bits else "unquantized"
    log_suffix = f"_{quantization_bits}bit" if quantization_bits else ""
    log_file_name = f"merged_delta_comparison{log_suffix}.log"

    print("=" * 80)
    print(f"Delta Compression Analysis: Merged Model vs Base Model ({quantization_label})")
    print("=" * 80)
    print(f"Base model: {base_model_name}")
    print(f"Merged model: {merged_model_path}")
    print(f"Quantization: {quantization_label}")
    print("=" * 80 + "\n")

    # Load base model
    print("Loading base model...")
    start_time = time.time()
    base_model = load_model(base_model_name, quantization_bits)
    print(f"Base model loaded in {time.time() - start_time:.2f}s\n")

    # Load merged model
    print("Loading merged model...")
    start_time = time.time()
    merged_model = load_model(merged_model_path, quantization_bits)
    print(f"Merged model loaded in {time.time() - start_time:.2f}s\n")

    # Calculate delta
    print("Calculating delta between models...")
    print("=" * 80 + "\n")
    delta_percentage = calculate_delta_nonzero_weights(base_model, merged_model)

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Quantization: {quantization_label}")
    print(f"Delta (percentage of nonzero weight differences): {delta_percentage:.6f}%")
    print("=" * 80)

    # Write to log file
    with open(log_file_name, "w") as log_file:
        log_file.write(f"Delta Compression Analysis: Merged Model vs Base Model ({quantization_label})\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Base model: {base_model_name}\n")
        log_file.write(f"Merged model: {merged_model_path}\n")
        log_file.write(f"Quantization: {quantization_label}\n")
        log_file.write(f"Delta percentage: {delta_percentage:.6f}%\n")
        log_file.write("=" * 80 + "\n")

    print(f"\nResults saved to: {log_file_name}\n")

    # Cleanup
    del base_model
    del merged_model
    torch.cuda.empty_cache()

    return delta_percentage


def run_checkpoint_comparison(checkpoint1_path, checkpoint2_path, quantization_bits, label):
    """
    Compare two checkpoints directly (e.g., sequential task comparisons).

    Args:
        checkpoint1_path: Path to the first checkpoint
        checkpoint2_path: Path to the second checkpoint
        quantization_bits: None for no quantization, 4 or 8 for quantized comparison
        label: Descriptive label for this comparison (e.g., "task_1 vs task_0")

    Returns:
        float: Delta percentage
    """
    quantization_label = f"{quantization_bits}-bit" if quantization_bits else "unquantized"
    print("=" * 80)
    print(f"Sequential Comparison: {label} ({quantization_label})")
    print("=" * 80)
    print(f"Checkpoint 1: {checkpoint1_path}")
    print(f"Checkpoint 2: {checkpoint2_path}")
    print(f"Quantization: {quantization_label}")
    print("=" * 80 + "\n")

    # Load first checkpoint
    print("Loading checkpoint 1...")
    start_time = time.time()
    checkpoint1 = load_model(checkpoint1_path, quantization_bits)
    print(f"Checkpoint 1 loaded in {time.time() - start_time:.2f}s\n")

    # Load second checkpoint
    print("Loading checkpoint 2...")
    start_time = time.time()
    checkpoint2 = load_model(checkpoint2_path, quantization_bits)
    print(f"Checkpoint 2 loaded in {time.time() - start_time:.2f}s\n")

    # Calculate delta
    print("Calculating delta between checkpoints...")
    print("=" * 80 + "\n")
    delta_percentage = calculate_delta_nonzero_weights(checkpoint1, checkpoint2)

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Comparison: {label}")
    print(f"Quantization: {quantization_label}")
    print(f"Delta (percentage of nonzero weight differences): {delta_percentage:.6f}%")
    print("=" * 80)

    # Cleanup
    del checkpoint1
    del checkpoint2
    torch.cuda.empty_cache()

    return delta_percentage


def run_continual_learning_comparison(base_model_name, checkpoints, quantization_bits, comparison_mode):
    """
    Run delta comparisons for continual learning checkpoints.

    Args:
        base_model_name: Name or path of the base model
        checkpoints: List of tuples [(task_id, task_name, checkpoint_path), ...]
        quantization_bits: None for no quantization, 4 or 8 for quantized comparison
        comparison_mode: "cumulative", "sequential", or "both"

    Returns:
        Dictionary containing results for cumulative and/or sequential comparisons
    """
    results = {"cumulative": {}, "sequential": {}}
    quantization_label = f"{quantization_bits}-bit" if quantization_bits else "unquantized"

    # Cumulative comparisons (each checkpoint vs base)
    if comparison_mode in ["cumulative", "both"]:
        print("\n" + "=" * 80)
        print(f"CUMULATIVE COMPARISONS (Each checkpoint vs base model) - {quantization_label}")
        print("=" * 80 + "\n")

        for task_id, task_name, checkpoint_path in checkpoints:
            checkpoint_name = f"task_{task_id}_{task_name}"
            try:
                print(f"\n{'='*80}")
                print(f"Comparing: {checkpoint_name} vs Base Model")
                print(f"{'='*80}\n")

                delta = run_comparison(base_model_name, checkpoint_path, quantization_bits)
                results["cumulative"][checkpoint_name] = {
                    "delta_percentage": delta,
                    "quantization": quantization_label,
                    "task_id": task_id,
                    "task_name": task_name
                }

                # Brief pause between comparisons
                time.sleep(2)

            except Exception as e:
                print(f"Error during cumulative comparison for {checkpoint_name}: {e}")
                results["cumulative"][checkpoint_name] = {
                    "error": str(e),
                    "quantization": quantization_label,
                    "task_id": task_id,
                    "task_name": task_name
                }

    # Sequential comparisons (task_N vs task_N-1)
    if comparison_mode in ["sequential", "both"]:
        print("\n" + "=" * 80)
        print(f"SEQUENTIAL COMPARISONS (Task N vs Task N-1) - {quantization_label}")
        print("=" * 80 + "\n")

        for i in range(1, len(checkpoints)):
            prev_task_id, prev_task_name, prev_checkpoint_path = checkpoints[i-1]
            curr_task_id, curr_task_name, curr_checkpoint_path = checkpoints[i]

            prev_name = f"task_{prev_task_id}_{prev_task_name}"
            curr_name = f"task_{curr_task_id}_{curr_task_name}"
            comparison_label = f"{curr_name} vs {prev_name}"

            try:
                print(f"\n{'='*80}")
                print(f"Comparing: {comparison_label}")
                print(f"{'='*80}\n")

                delta = run_checkpoint_comparison(
                    prev_checkpoint_path,
                    curr_checkpoint_path,
                    quantization_bits,
                    comparison_label
                )

                results["sequential"][comparison_label] = {
                    "delta_percentage": delta,
                    "quantization": quantization_label,
                    "from_task_id": prev_task_id,
                    "to_task_id": curr_task_id
                }

                # Brief pause between comparisons
                time.sleep(2)

            except Exception as e:
                print(f"Error during sequential comparison for {comparison_label}: {e}")
                results["sequential"][comparison_label] = {
                    "error": str(e),
                    "quantization": quantization_label,
                    "from_task_id": prev_task_id,
                    "to_task_id": curr_task_id
                }

    return results


def main():
    """
    Main function for comparing merged model with base model.
    """
    parser = argparse.ArgumentParser(
        description="Compare delta compression for continual learning checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all checkpoints with both cumulative and sequential analysis
  python compare_merged_delta.py --task C-STANCE --task FOMC --task MeetingBank

  # Only sequential comparisons with 4-bit quantization
  python compare_merged_delta.py --task C-STANCE --task FOMC --comparison-mode sequential --quantization 4bit

  # Cumulative comparisons only (each checkpoint vs base)
  python compare_merged_delta.py --task C-STANCE --task FOMC --comparison-mode cumulative
        """
    )
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        help="Task names in continual learning sequence (can be passed multiple times)"
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default="./merged_models",
        help="Base directory containing merged model checkpoints (default: ./merged_models)"
    )
    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=["cumulative", "sequential", "both"],
        default="both",
        help="Comparison mode: 'cumulative' (each vs base), 'sequential' (task N vs N-1), or 'both' (default: both)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "4bit", "8bit", "all"],
        default="all",
        help="Quantization level: 'none' (unquantized), '4bit', '8bit', or 'all' (run all comparisons, default)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help=f"Base model name or path (default: {BASE_MODEL})"
    )

    args = parser.parse_args()

    base_model_name = args.base_model
    task_sequence = args.task
    checkpoint_base_dir = args.checkpoint_base_dir
    comparison_mode = args.comparison_mode

    # Validate checkpoint directory
    if not Path(checkpoint_base_dir).exists():
        print(f"Error: Checkpoint directory does not exist: {checkpoint_base_dir}")
        return

    # Discover checkpoints
    print("=" * 80)
    print("DISCOVERING CHECKPOINTS")
    print("=" * 80)
    print(f"Task sequence: {task_sequence}")
    print(f"Checkpoint directory: {checkpoint_base_dir}")
    print("=" * 80 + "\n")

    try:
        checkpoints = find_checkpoints(checkpoint_base_dir, task_sequence, use_merged_models=False)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    print(f"\nFound {len(checkpoints)} checkpoints:")
    for task_id, task_name, checkpoint_path in checkpoints:
        print(f"  - task_{task_id}_{task_name}: {checkpoint_path}")
    print()

    # Determine which quantization levels to run
    comparisons = []
    if args.quantization == "all":
        comparisons = [None, 4, 8]  # None = unquantized
    elif args.quantization == "none":
        comparisons = [None]
    elif args.quantization == "4bit":
        comparisons = [4]
    elif args.quantization == "8bit":
        comparisons = [8]

    # Run comparisons for each quantization level
    all_results = {}
    for quant_bits in comparisons:
        quant_label = f"{quant_bits}-bit" if quant_bits else "unquantized"
        print(f"\n{'=' * 80}")
        print(f"Starting comparisons: {quant_label}")
        print(f"{'=' * 80}\n")

        try:
            results = run_continual_learning_comparison(
                base_model_name,
                checkpoints,
                quant_bits,
                comparison_mode
            )
            all_results[quant_label] = results
        except Exception as e:
            print(f"Error during {quant_label} comparisons: {e}")
            all_results[quant_label] = {"error": str(e)}

        # Add spacing between quantization levels
        if quant_bits != comparisons[-1]:
            print("\n" + "=" * 80)
            print("Clearing memory before next quantization level...")
            print("=" * 80 + "\n")
            time.sleep(2)

    # Save results to JSON files
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    for quant_label, results in all_results.items():
        if "error" in results:
            continue

        # Save cumulative results
        if results.get("cumulative"):
            filename = f"delta_comparison_cumulative_{quant_label.replace('-bit', 'bit').replace('unquantized', 'unquantized')}.json"
            with open(filename, 'w') as f:
                json.dump(results["cumulative"], f, indent=2)
            print(f"Saved cumulative results: {filename}")

        # Save sequential results
        if results.get("sequential"):
            filename = f"delta_comparison_sequential_{quant_label.replace('-bit', 'bit').replace('unquantized', 'unquantized')}.json"
            with open(filename, 'w') as f:
                json.dump(results["sequential"], f, indent=2)
            print(f"Saved sequential results: {filename}")

    print("=" * 80 + "\n")

    # Print enhanced summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80 + "\n")

    for quant_label, results in all_results.items():
        if "error" in results:
            print(f"{quant_label}: Failed - {results['error']}\n")
            continue

        print(f"{'=' * 80}")
        print(f"Quantization: {quant_label}")
        print(f"{'=' * 80}")

        # Cumulative results table
        if results.get("cumulative"):
            print("\nCumulative Comparisons (Each checkpoint vs Base Model):")
            print(f"{'Checkpoint':<40} | {'Delta vs Base'}")
            print("-" * 80)

            cumulative_deltas = []
            for checkpoint_name, data in sorted(results["cumulative"].items()):
                if "error" not in data:
                    delta = data["delta_percentage"]
                    cumulative_deltas.append(delta)
                    print(f"{checkpoint_name:<40} | {delta:>12.6f}%")
                else:
                    print(f"{checkpoint_name:<40} | {'ERROR':>12}")

            if cumulative_deltas:
                avg_cumulative = sum(cumulative_deltas) / len(cumulative_deltas)
                print("-" * 80)
                print(f"{'Average Cumulative Delta':<40} | {avg_cumulative:>12.6f}%")

        # Sequential results table
        if results.get("sequential"):
            print("\nSequential Comparisons (Task N vs Task N-1):")
            print(f"{'Comparison':<60} | {'Delta'}")
            print("-" * 80)

            sequential_deltas = []
            for comparison_label, data in sorted(results["sequential"].items()):
                if "error" not in data:
                    delta = data["delta_percentage"]
                    sequential_deltas.append(delta)
                    print(f"{comparison_label:<60} | {delta:>12.6f}%")
                else:
                    print(f"{comparison_label:<60} | {'ERROR':>12}")

            if sequential_deltas:
                avg_sequential = sum(sequential_deltas) / len(sequential_deltas)
                max_sequential = max(sequential_deltas)
                print("-" * 80)
                print(f"{'Average Incremental Delta':<60} | {avg_sequential:>12.6f}%")
                print(f"{'Largest Single-Task Delta':<60} | {max_sequential:>12.6f}%")

        print()

    print("=" * 80)
    print("ALL COMPARISONS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
