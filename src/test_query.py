"""
Interactive Query Testing

Test your checkpointed model with custom queries.

Usage:
    # Test a checkpoint
    python test_query.py --checkpoint ./lora-continual/task_0_C-STANCE
    
    # Single query (non-interactive)
    python test_query.py --checkpoint ./lora-continual/task_0_C-STANCE --query "Your question here"
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent / 'src'))
from constants import MAX_PROMPT_LEN, MAX_ANS_LEN
from model_loader import ModelLoader


def generate(model, tokenizer, query, max_tokens=MAX_ANS_LEN, temperature=0.1):
    """Generate response for query."""

    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT_LEN)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.unk_token_id,
            temperature=temperature,
            do_sample=True,
            use_cache=True
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    return full_text, generated_text


def interactive_mode(model, tokenizer, max_tokens=256, temperature=0.1):
    """Interactive query loop."""

    print("=" * 80)
    print("INTERACTIVE MODE")
    print("Type your query and press Enter. Type 'quit' to exit.")
    print("=" * 80)

    while True:
        print("\n> ", end="")
        query = input()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Bye!")
            break

        if not query.strip():
            continue

        print("\nGenerating...")
        full_text, generated_text = generate(model, tokenizer, query, max_tokens, temperature)

        print("\n" + "=" * 80)
        print("GENERATED TEXT ONLY:")
        print("=" * 80)
        print(generated_text)
        print("\n" + "=" * 80)
        print("FULL OUTPUT (Prompt + Generated):")
        print("=" * 80)
        print(full_text)
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test model with queries")

    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--base-model", action="store_true", help="Use base model only")
    parser.add_argument("--query", type=str, help="Single query (non-interactive)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature (default: 0.1)")
    parser.add_argument("--no-quantization", action="store_true", help="Disable quantization", default=True)

    args = parser.parse_args()

    # Validate args
    if not args.checkpoint and not args.base_model:
        parser.error("Must specify --checkpoint or --base-model")

    # Load model using ModelLoader
    loader = ModelLoader(use_quantization=not args.no_quantization)
    checkpoint = None if args.base_model else args.checkpoint
    model, tokenizer = loader.load_model_and_tokenizer(checkpoint_path=checkpoint, eval_mode=True)

    # Single query or interactive
    if args.query:
        print(f"Query: {args.query}\n")
        full_text, generated_text = generate(model, tokenizer, args.query, args.max_tokens, args.temperature)

        print("=" * 80)
        print("GENERATED TEXT ONLY:")
        print("=" * 80)
        print(generated_text)
        print("\n" + "=" * 80)
        print("FULL OUTPUT (Prompt + Generated):")
        print("=" * 80)
        print(full_text)
        print("=" * 80)
    else:
        interactive_mode(model, tokenizer, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
