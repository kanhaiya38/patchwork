"""
Multi-Task Delta Compression Evaluation for LoRA Continual Learning

Evaluates compression across all available task transitions, similar to delta_cmp.py
but with comprehensive compression analysis.
"""

import os
import json
import time
import torch
import psutil
import numpy as np
from pathlib import Path
from safetensors.torch import load_file, save_file
from typing import Dict, Tuple, Any, List
import tempfile
import pandas as pd


class MultiTaskCompressionEvaluator:
    """Evaluates compression across multiple task transitions."""

    def __init__(self, continual_dir: str):
        """
        Args:
            continual_dir: Path to continual learning checkpoint directory
        """
        self.continual_dir = Path(continual_dir)
        self.tasks = self._discover_tasks()
        self.task_pairs = self._create_task_pairs()

        print("=" * 70)
        print("Multi-Task Compression Evaluation")
        print("=" * 70)
        print(f"\nContinual directory: {self.continual_dir}")
        print(f"Discovered tasks: {len(self.tasks)}")
        for i, task in enumerate(self.tasks):
            print(f"  {i}. {task}")
        print(f"\nTask pairs to evaluate: {len(self.task_pairs)}")
        for base, target in self.task_pairs:
            print(f"  {base} → {target}")
        print()

    def _discover_tasks(self) -> List[str]:
        """Discover available task checkpoints in continual directory."""
        tasks = []
        if not self.continual_dir.exists():
            print(f"Warning: {self.continual_dir} does not exist!")
            return tasks

        # Find all task directories
        for item in sorted(self.continual_dir.iterdir()):
            if item.is_dir() and item.name.startswith('task_'):
                # Check if it has adapter_model.safetensors
                if (item / "adapter_model.safetensors").exists():
                    tasks.append(item.name)

        return tasks

    def _create_task_pairs(self) -> List[Tuple[str, str]]:
        """Create list of task pairs to evaluate (consecutive + first vs last)."""
        pairs = []

        # Consecutive pairs
        for i in range(len(self.tasks) - 1):
            pairs.append((self.tasks[i], self.tasks[i + 1]))

        # First vs last (if more than 2 tasks)
        if len(self.tasks) > 2:
            pairs.append((self.tasks[0], self.tasks[-1]))

        return pairs

    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

    def load_checkpoint(self, task_name: str) -> Tuple[Dict[str, torch.Tensor], int]:
        """Load checkpoint and return weights + file size."""
        checkpoint_path = self.continual_dir / task_name / "adapter_model.safetensors"
        weights = load_file(str(checkpoint_path))
        file_size = os.path.getsize(checkpoint_path)
        return weights, file_size

    def compute_delta(self, base_weights: Dict[str, torch.Tensor],
                     target_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute weight differences between target and base."""
        delta = {}
        for key in target_weights.keys():
            if key in base_weights:
                delta[key] = target_weights[key] - base_weights[key]
            else:
                delta[key] = target_weights[key]
        return delta

    def compute_delta_statistics(self, delta: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute statistics about the delta."""
        total_params = sum(t.numel() for t in delta.values())
        nonzero_params = sum((t != 0).sum().item() for t in delta.values())

        # Compute magnitude statistics
        all_deltas = torch.cat([t.flatten() for t in delta.values()])

        return {
            "total_params": total_params,
            "nonzero_params": nonzero_params,
            "zero_params": total_params - nonzero_params,
            "sparsity_pct": (1 - nonzero_params / total_params) * 100,
            "delta_mean": all_deltas.mean().item(),
            "delta_std": all_deltas.std().item(),
            "delta_min": all_deltas.min().item(),
            "delta_max": all_deltas.max().item(),
            "delta_abs_mean": all_deltas.abs().mean().item(),
        }

    def verify_reconstruction(self, original: Dict[str, torch.Tensor],
                            reconstructed: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Verify reconstructed weights match original."""
        max_error = 0.0
        avg_error = 0.0
        total_elements = 0

        for key in original.keys():
            if key in reconstructed:
                diff = torch.abs(original[key] - reconstructed[key])
                max_error = max(max_error, diff.max().item())
                avg_error += diff.sum().item()
                total_elements += diff.numel()

        avg_error /= total_elements if total_elements > 0 else 1

        return {
            "max_absolute_error": max_error,
            "mean_absolute_error": avg_error,
            "l2_norm": np.sqrt(avg_error)
        }

    def evaluate_fp16_compression(self, base_weights: Dict[str, torch.Tensor],
                                  target_weights: Dict[str, torch.Tensor],
                                  base_size: int, target_size: int) -> Dict[str, Any]:
        """Evaluate FP16 quantized delta compression."""
        results = {"method": "FP16"}

        # Compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        delta = self.compute_delta(base_weights, target_weights)
        quantized_delta = {k: v.to(torch.float16) for k, v in delta.items()}

        with tempfile.TemporaryDirectory() as tmp_dir:
            delta_path = Path(tmp_dir) / "delta_fp16.safetensors"
            save_file(quantized_delta, str(delta_path))
            compressed_size = os.path.getsize(delta_path)

            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded = load_file(str(delta_path))
            reconstructed = {}
            for key, delta_val in loaded.items():
                if key in base_weights:
                    reconstructed[key] = base_weights[key] + delta_val.float()
                else:
                    reconstructed[key] = delta_val.float()

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Metrics
        results["compressed_size_mb"] = compressed_size / 1024**2
        results["total_size_mb"] = (base_size + compressed_size) / 1024**2
        total_original = base_size + target_size
        total_compressed = base_size + compressed_size
        results["space_saved_mb"] = (total_original - total_compressed) / 1024**2
        results["compression_ratio"] = total_original / total_compressed
        results["space_saved_pct"] = (1 - total_compressed / total_original) * 100

        # Accuracy
        accuracy = self.verify_reconstruction(target_weights, reconstructed)
        results.update(accuracy)

        return results

    def evaluate_int8_compression(self, base_weights: Dict[str, torch.Tensor],
                                  target_weights: Dict[str, torch.Tensor],
                                  base_size: int, target_size: int) -> Dict[str, Any]:
        """Evaluate INT8 quantized delta compression."""
        results = {"method": "INT8"}

        # Compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        delta = self.compute_delta(base_weights, target_weights)
        quantized_delta = {}

        for key, tensor in delta.items():
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            scale = (tensor_max - tensor_min) / 255.0
            zero_point = -torch.round(tensor_min / scale).to(torch.int8)
            quantized = torch.round(tensor / scale + zero_point.float()).clamp(-128, 127).to(torch.int8)

            quantized_delta[key] = quantized
            quantized_delta[f"{key}_scale"] = scale.reshape(1)
            quantized_delta[f"{key}_zero_point"] = zero_point.reshape(1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            delta_path = Path(tmp_dir) / "delta_int8.safetensors"
            save_file(quantized_delta, str(delta_path))
            compressed_size = os.path.getsize(delta_path)

            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded = load_file(str(delta_path))
            reconstructed = {}

            for key in delta.keys():
                quantized = loaded[key]
                scale = loaded[f"{key}_scale"]
                zero_point = loaded[f"{key}_zero_point"]
                dequantized = (quantized.float() - zero_point.float()) * scale

                if key in base_weights:
                    reconstructed[key] = base_weights[key] + dequantized
                else:
                    reconstructed[key] = dequantized

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Metrics
        results["compressed_size_mb"] = compressed_size / 1024**2
        results["total_size_mb"] = (base_size + compressed_size) / 1024**2
        total_original = base_size + target_size
        total_compressed = base_size + compressed_size
        results["space_saved_mb"] = (total_original - total_compressed) / 1024**2
        results["compression_ratio"] = total_original / total_compressed
        results["space_saved_pct"] = (1 - total_compressed / total_original) * 100

        # Accuracy
        accuracy = self.verify_reconstruction(target_weights, reconstructed)
        results.update(accuracy)

        return results

    def evaluate_pair(self, base_task: str, target_task: str) -> Dict[str, Any]:
        """Evaluate compression for a single task pair."""
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {base_task} → {target_task}")
        print('=' * 70)

        # Load checkpoints
        base_weights, base_size = self.load_checkpoint(base_task)
        target_weights, target_size = self.load_checkpoint(target_task)

        print(f"Base checkpoint:   {base_size / 1024**2:.2f} MB")
        print(f"Target checkpoint: {target_size / 1024**2:.2f} MB")
        print(f"Total:             {(base_size + target_size) / 1024**2:.2f} MB")

        # Compute delta statistics
        delta = self.compute_delta(base_weights, target_weights)
        delta_stats = self.compute_delta_statistics(delta)

        print(f"\nDelta statistics:")
        print(f"  Sparsity: {delta_stats['sparsity_pct']:.4f}%")
        print(f"  Nonzero changes: {delta_stats['nonzero_params']:,} / {delta_stats['total_params']:,}")
        print(f"  Mean delta: {delta_stats['delta_mean']:.6f}")
        print(f"  Std delta:  {delta_stats['delta_std']:.6f}")
        print(f"  Abs mean:   {delta_stats['delta_abs_mean']:.6f}")

        # Evaluate compression methods
        results = {
            "base_task": base_task,
            "target_task": target_task,
            "base_size_mb": base_size / 1024**2,
            "target_size_mb": target_size / 1024**2,
            "delta_stats": delta_stats,
        }

        print("\nEvaluating FP16 compression...")
        results["fp16"] = self.evaluate_fp16_compression(base_weights, target_weights, base_size, target_size)

        print("Evaluating INT8 compression...")
        results["int8"] = self.evaluate_int8_compression(base_weights, target_weights, base_size, target_size)

        # Cleanup
        del base_weights, target_weights, delta
        torch.cuda.empty_cache()

        return results

    def run_evaluation(self) -> List[Dict[str, Any]]:
        """Run evaluation on all task pairs."""
        all_results = []

        for base_task, target_task in self.task_pairs:
            results = self.evaluate_pair(base_task, target_task)
            all_results.append(results)

        return all_results

    def generate_summary_report(self, all_results: List[Dict[str, Any]]):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 70)
        print("MULTI-TASK COMPRESSION SUMMARY")
        print("=" * 70)

        # Separate consecutive and first-vs-last
        consecutive = [r for r in all_results if not (r['base_task'] == self.tasks[0] and r['target_task'] == self.tasks[-1])]
        first_vs_last = [r for r in all_results if r['base_task'] == self.tasks[0] and r['target_task'] == self.tasks[-1]]

        print("\n" + "-" * 70)
        print("CONSECUTIVE TASK TRANSITIONS")
        print("-" * 70)

        # Table header
        print(f"\n{'Transition':<35} {'Sparsity %':<12} {'FP16 Saved':<15} {'INT8 Saved':<15}")
        print("-" * 70)

        for result in consecutive:
            transition = f"{result['base_task']} → {result['target_task']}"
            sparsity = result['delta_stats']['sparsity_pct']
            fp16_saved = result['fp16']['space_saved_pct']
            int8_saved = result['int8']['space_saved_pct']
            print(f"{transition:<35} {sparsity:<12.4f} {fp16_saved:<15.2f} {int8_saved:<15.2f}")

        # Average stats for consecutive
        if consecutive:
            avg_sparsity = np.mean([r['delta_stats']['sparsity_pct'] for r in consecutive])
            avg_fp16 = np.mean([r['fp16']['space_saved_pct'] for r in consecutive])
            avg_int8 = np.mean([r['int8']['space_saved_pct'] for r in consecutive])
            print("-" * 70)
            print(f"{'AVERAGE':<35} {avg_sparsity:<12.4f} {avg_fp16:<15.2f} {avg_int8:<15.2f}")

        # First vs last
        if first_vs_last:
            print("\n" + "-" * 70)
            print("FIRST VS LAST COMPARISON")
            print("-" * 70)
            result = first_vs_last[0]
            transition = f"{result['base_task']} → {result['target_task']}"
            sparsity = result['delta_stats']['sparsity_pct']
            fp16_saved = result['fp16']['space_saved_pct']
            int8_saved = result['int8']['space_saved_pct']
            print(f"{transition:<35} {sparsity:<12.4f} {fp16_saved:<15.2f} {int8_saved:<15.2f}")

        # Error analysis
        print("\n" + "-" * 70)
        print("ERROR ANALYSIS (MAX ABSOLUTE ERROR)")
        print("-" * 70)

        print(f"\n{'Transition':<35} {'FP16 Error':<15} {'INT8 Error':<15}")
        print("-" * 70)

        for result in consecutive:
            transition = f"{result['base_task']} → {result['target_task']}"
            fp16_error = result['fp16']['max_absolute_error']
            int8_error = result['int8']['max_absolute_error']
            print(f"{transition:<35} {fp16_error:<15.2e} {int8_error:<15.2e}")

        # Delta magnitude analysis
        print("\n" + "-" * 70)
        print("DELTA MAGNITUDE ANALYSIS")
        print("-" * 70)

        print(f"\n{'Transition':<35} {'Mean |Δ|':<15} {'Std |Δ|':<15}")
        print("-" * 70)

        for result in consecutive:
            transition = f"{result['base_task']} → {result['target_task']}"
            abs_mean = result['delta_stats']['delta_abs_mean']
            std = result['delta_stats']['delta_std']
            print(f"{transition:<35} {abs_mean:<15.6f} {std:<15.6f}")

        # Projection for 8 tasks
        print("\n" + "=" * 70)
        print("PROJECTION FOR 8 TASKS")
        print("=" * 70)

        if consecutive:
            avg_checkpoint_size = np.mean([r['base_size_mb'] for r in all_results])
            avg_fp16_compressed = np.mean([r['fp16']['compressed_size_mb'] for r in consecutive])
            avg_int8_compressed = np.mean([r['int8']['compressed_size_mb'] for r in consecutive])

            baseline_8_tasks = avg_checkpoint_size * 8
            fp16_8_tasks = avg_checkpoint_size + (avg_fp16_compressed * 7)
            int8_8_tasks = avg_checkpoint_size + (avg_int8_compressed * 7)

            print(f"\nAssuming average checkpoint size: {avg_checkpoint_size:.2f} MB")
            print(f"\nBaseline (no compression):     {baseline_8_tasks:.2f} MB")
            print(f"FP16 compression:              {fp16_8_tasks:.2f} MB (saves {baseline_8_tasks - fp16_8_tasks:.2f} MB, {(1 - fp16_8_tasks/baseline_8_tasks)*100:.1f}%)")
            print(f"INT8 compression:              {int8_8_tasks:.2f} MB (saves {baseline_8_tasks - int8_8_tasks:.2f} MB, {(1 - int8_8_tasks/baseline_8_tasks)*100:.1f}%)")


def main():
    """Main evaluation function."""
    continual_dir = "./run_lora_extended_complete/continual"

    print("=" * 70)
    print("Multi-Task Delta Compression Evaluation")
    print("For LoRA Continual Learning Checkpoints")
    print("=" * 70)
    print()

    # Create evaluator
    evaluator = MultiTaskCompressionEvaluator(continual_dir)

    if len(evaluator.tasks) < 2:
        print("Error: Need at least 2 tasks to evaluate compression!")
        return

    # Run evaluation
    all_results = evaluator.run_evaluation()

    # Generate summary report
    evaluator.generate_summary_report(all_results)

    # Save detailed results
    output_file = "compression_multitask_results.json"

    # Convert to JSON-serializable format
    json_results = []
    for result in all_results:
        json_result = {
            "base_task": result["base_task"],
            "target_task": result["target_task"],
            "base_size_mb": float(result["base_size_mb"]),
            "target_size_mb": float(result["target_size_mb"]),
            "delta_stats": {k: float(v) for k, v in result["delta_stats"].items()},
            "fp16": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in result["fp16"].items()},
            "int8": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in result["int8"].items()},
        }
        json_results.append(json_result)

    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
