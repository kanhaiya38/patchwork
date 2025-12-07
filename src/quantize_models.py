import torch
import time
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_quantization_config(bits):
    """
    Get BitsAndBytesConfig for quantization.

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


def quantize_and_save_model(input_path, output_path, quantization_bits):
    """
    Load a model with quantization and save it to disk.

    Args:
        input_path: Path to the input model
        output_path: Path to save the quantized model
        quantization_bits: 4 or 8 for quantization bits

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\nProcessing: {input_path}")
        print(f"  Output: {output_path}")
        print(f"  Quantization: {quantization_bits}-bit")

        # Create output directory
        Path(output_path).mkdir(parents=True, exist_ok=True)

        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(input_path, use_fast=False)

        print("  Loading model with quantization...")
        start_time = time.time()
        quantization_config = get_quantization_config(quantization_bits)
        model = AutoModelForCausalLM.from_pretrained(
            input_path,
            quantization_config=quantization_config,
            device_map="auto"
        )
        load_time = time.time() - start_time
        print(f"  Model loaded in {load_time:.2f}s")

        print("  Saving quantized model and tokenizer...")
        start_time = time.time()
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Remove quantization_config from config.json to avoid loading issues
        import json
        config_path = Path(output_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            if 'quantization_config' in config:
                del config['quantization_config']
                print("  Removing quantization_config from config.json...")
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

        save_time = time.time() - start_time
        print(f"  Model and tokenizer saved in {save_time:.2f}s")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        print(f"  Success! Total time: {load_time + save_time:.2f}s")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def discover_models(input_dir, pattern=None):
    """
    Discover model directories in the input directory.

    Args:
        input_dir: Base directory containing models
        pattern: Optional glob pattern to filter models

    Returns:
        List of model directory paths
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Find all subdirectories that contain model files (config.json or pytorch_model.bin)
    model_dirs = []

    if pattern:
        # Use glob pattern
        for candidate in input_path.glob(pattern):
            if candidate.is_dir() and (
                (candidate / "config.json").exists() or
                (candidate / "pytorch_model.bin").exists() or
                any(candidate.glob("*.safetensors"))
            ):
                model_dirs.append(candidate)
    else:
        # Search all subdirectories
        for candidate in input_path.rglob("config.json"):
            model_dir = candidate.parent
            if model_dir not in model_dirs:
                model_dirs.append(model_dir)

    return sorted(model_dirs)


def main():
    """
    Main function for quantizing and saving models.
    """
    parser = argparse.ArgumentParser(
        description="Quantize merged models and save them to disk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize all models in ./merged_models to 4-bit
  python quantize_models.py --input-dir ./merged_models --output-dir ./quantized_models
        """
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./merged_models",
        help="Input directory containing merged models (default: ./merged_models)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./quantized_models",
        help="Output directory for quantized models (default: ./quantized_models)"
    )
    parser.add_argument(
        "--quantization",
        type=int,
        choices=[4, 8],
        default=4,
        help="Quantization bits: 4 or 8 (default: 4)"
    )
    parser.add_argument(
        "--model-pattern",
        type=str,
        default=None,
        help="Optional glob pattern to filter models (e.g., 'task_*')"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MODEL QUANTIZATION SCRIPT")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quantization: {args.quantization}-bit")
    if args.model_pattern:
        print(f"Model pattern: {args.model_pattern}")
    print("=" * 80)

    # Discover models
    try:
        print("\nDiscovering models...")
        models = discover_models(args.input_dir, args.model_pattern)
        print(f"Found {len(models)} model(s) to quantize:")
        for model_path in models:
            print(f"  - {model_path}")
    except Exception as e:
        print(f"Error discovering models: {e}")
        return

    if not models:
        print("\nNo models found. Exiting.")
        return

    # Process each model
    print("\n" + "=" * 80)
    print("QUANTIZING MODELS")
    print("=" * 80)

    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)

    success_count = 0
    failure_count = 0

    for model_path in models:
        try:
            relative_path = model_path.relative_to(input_base)
        except ValueError:
            relative_path = model_path.name

        output_path = output_base / relative_path

        if quantize_and_save_model(str(model_path), str(output_path), args.quantization):
            success_count += 1
        else:
            failure_count += 1

        if model_path != models[-1]:
            print("\nPausing before next model...")
            time.sleep(2)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total models processed: {len(models)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()
