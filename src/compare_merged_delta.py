import torch
import time
import argparse
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from constants import BASE_MODEL


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


def main():
    """
    Main function for comparing merged model with base model.
    """
    parser = argparse.ArgumentParser(
        description="Compare delta compression between merged and base models with optional quantization"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "4bit", "8bit", "all"],
        default="all",
        help="Quantization level: 'none' (unquantized), '4bit', '8bit', or 'all' (run all comparisons)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help=f"Base model name or path (default: {BASE_MODEL})"
    )
    parser.add_argument(
        "--merged-model",
        type=str,
        default="merged-openllama-3b",
        help="Merged model path (default: merged-openllama-3b)"
    )

    args = parser.parse_args()

    base_model_name = args.base_model
    merged_model_path = args.merged_model

    # Determine which comparisons to run
    comparisons = []
    if args.quantization == "all":
        comparisons = [None, 4, 8]  # None = unquantized
    elif args.quantization == "none":
        comparisons = [None]
    elif args.quantization == "4bit":
        comparisons = [4]
    elif args.quantization == "8bit":
        comparisons = [8]

    # Run comparisons
    results = {}
    for quant_bits in comparisons:
        quant_label = f"{quant_bits}-bit" if quant_bits else "unquantized"
        print(f"\n{'=' * 80}")
        print(f"Starting comparison: {quant_label}")
        print(f"{'=' * 80}\n")

        try:
            delta = run_comparison(base_model_name, merged_model_path, quant_bits)
            results[quant_label] = delta
        except Exception as e:
            print(f"Error during {quant_label} comparison: {e}")
            results[quant_label] = None

        # Add some spacing between comparisons
        if quant_bits != comparisons[-1]:
            print("\n" + "=" * 80)
            print("Clearing memory before next comparison...")
            print("=" * 80 + "\n")
            time.sleep(2)  # Brief pause to ensure cleanup

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL COMPARISONS")
    print("=" * 80)
    for quant_label, delta in results.items():
        if delta is not None:
            print(f"{quant_label:>15}: {delta:.6f}%")
        else:
            print(f"{quant_label:>15}: Failed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
