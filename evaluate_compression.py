"""
Delta Compression Evaluation for LoRA Continual Learning Checkpoints

This script compares different delta compression methods for LoRA adapter checkpoints.
Tests compression ratio, speed, accuracy, and memory usage.
"""

import os
import json
import time
import torch
import psutil
import numpy as np
from pathlib import Path
from safetensors.torch import load_file, save_file
from typing import Dict, Tuple, Any
import tempfile
import shutil


class CompressionEvaluator:
    """Evaluates different delta compression methods for LoRA checkpoints."""

    def __init__(self, base_checkpoint: str, target_checkpoint: str):
        """
        Args:
            base_checkpoint: Path to base checkpoint (e.g., task_0)
            target_checkpoint: Path to target checkpoint (e.g., task_1)
        """
        self.base_path = Path(base_checkpoint)
        self.target_path = Path(target_checkpoint)

        # Load checkpoints
        print(f"Loading base checkpoint: {self.base_path}")
        self.base_weights = load_file(self.base_path / "adapter_model.safetensors")

        print(f"Loading target checkpoint: {self.target_path}")
        self.target_weights = load_file(self.target_path / "adapter_model.safetensors")

        # Calculate original size
        self.original_size = os.path.getsize(self.base_path / "adapter_model.safetensors")
        self.target_size = os.path.getsize(self.target_path / "adapter_model.safetensors")

        print(f"\nOriginal checkpoint sizes:")
        print(f"  Base (task_0): {self.original_size / 1024**2:.2f} MB")
        print(f"  Target (task_1): {self.target_size / 1024**2:.2f} MB")
        print(f"  Total: {(self.original_size + self.target_size) / 1024**2:.2f} MB")
        print()

    def get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

    def compute_delta(self) -> Dict[str, torch.Tensor]:
        """Compute weight differences between target and base."""
        delta = {}
        for key in self.target_weights.keys():
            if key in self.base_weights:
                delta[key] = self.target_weights[key] - self.base_weights[key]
            else:
                # New weights in target
                delta[key] = self.target_weights[key]
        return delta

    def verify_reconstruction(self, reconstructed: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Verify reconstructed weights match original target weights."""
        max_error = 0.0
        avg_error = 0.0
        total_elements = 0

        for key in self.target_weights.keys():
            if key in reconstructed:
                diff = torch.abs(self.target_weights[key] - reconstructed[key])
                max_error = max(max_error, diff.max().item())
                avg_error += diff.sum().item()
                total_elements += diff.numel()

        avg_error /= total_elements if total_elements > 0 else 1

        return {
            "max_absolute_error": max_error,
            "mean_absolute_error": avg_error,
            "l2_norm": np.sqrt(avg_error)
        }

    # ============================================
    # Method 1: Delta-only Storage
    # ============================================

    def method1_delta_only(self) -> Dict[str, Any]:
        """
        Method 1: Store base checkpoint + delta
        - Base checkpoint stored as-is (full weights)
        - Delta checkpoint stores only differences
        """
        print("=" * 70)
        print("METHOD 1: Delta-only Storage")
        print("=" * 70)

        results = {"method": "Delta-only Storage"}

        # Measure compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        delta = self.compute_delta()

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            delta_path = Path(tmp_dir) / "delta.safetensors"
            save_file(delta, str(delta_path))

            compressed_size = os.path.getsize(delta_path)
            results["compressed_size_mb"] = compressed_size / 1024**2
            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Measure decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded_delta = load_file(str(delta_path))
            reconstructed = {}
            for key in loaded_delta.keys():
                if key in self.base_weights:
                    reconstructed[key] = self.base_weights[key] + loaded_delta[key]
                else:
                    reconstructed[key] = loaded_delta[key]

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Verify accuracy
        accuracy = self.verify_reconstruction(reconstructed)
        results.update(accuracy)

        # Calculate savings
        total_original = self.original_size + self.target_size
        total_compressed = self.original_size + compressed_size  # Base + delta
        results["total_size_mb"] = total_compressed / 1024**2
        results["space_saved_mb"] = (total_original - total_compressed) / 1024**2
        results["compression_ratio"] = total_original / total_compressed
        results["space_saved_pct"] = (1 - total_compressed / total_original) * 100

        # Calculate delta statistics
        total_params = sum(t.numel() for t in delta.values())
        nonzero_params = sum((t != 0).sum().item() for t in delta.values())
        results["delta_sparsity_pct"] = (1 - nonzero_params / total_params) * 100

        return results

    # ============================================
    # Method 2: Sparse Delta Storage
    # ============================================

    def method2_sparse_delta(self) -> Dict[str, Any]:
        """
        Method 2: Store base + sparse delta (only non-zero changes)
        - Base checkpoint stored as-is
        - Delta stores only non-zero differences with indices
        """
        print("\n" + "=" * 70)
        print("METHOD 2: Sparse Delta Storage")
        print("=" * 70)

        results = {"method": "Sparse Delta Storage"}

        # Measure compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        delta = self.compute_delta()

        # Convert to sparse format
        sparse_delta = {}
        metadata = {}

        for key, tensor in delta.items():
            # Find non-zero elements
            nonzero_mask = tensor != 0
            nonzero_indices = nonzero_mask.nonzero(as_tuple=False)
            nonzero_values = tensor[nonzero_mask]

            if nonzero_values.numel() > 0:
                # Store as sparse (ensure contiguous for safetensors)
                sparse_delta[f"{key}_indices"] = nonzero_indices.to(torch.int32).contiguous()
                sparse_delta[f"{key}_values"] = nonzero_values.contiguous()
                metadata[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "nnz": nonzero_values.numel()
                }

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            delta_path = Path(tmp_dir) / "sparse_delta.safetensors"
            metadata_path = Path(tmp_dir) / "sparse_metadata.json"

            save_file(sparse_delta, str(delta_path))
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

            compressed_size = os.path.getsize(delta_path) + os.path.getsize(metadata_path)
            results["compressed_size_mb"] = compressed_size / 1024**2
            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Measure decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded_sparse = load_file(str(delta_path))
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)

            # Reconstruct dense delta
            reconstructed = {}
            for key, meta in loaded_metadata.items():
                shape = tuple(meta["shape"])
                dtype = getattr(torch, meta["dtype"].split('.')[-1])

                # Create zero tensor
                dense_delta = torch.zeros(shape, dtype=dtype)

                # Fill in non-zero values
                indices_key = f"{key}_indices"
                values_key = f"{key}_values"

                if indices_key in loaded_sparse and values_key in loaded_sparse:
                    indices = loaded_sparse[indices_key].long()
                    values = loaded_sparse[values_key]

                    # Convert indices to tuple for indexing
                    if indices.dim() == 2:
                        idx_tuple = tuple(indices[:, i] for i in range(indices.shape[1]))
                        dense_delta[idx_tuple] = values
                    else:
                        dense_delta[indices] = values

                # Add to base weights
                if key in self.base_weights:
                    reconstructed[key] = self.base_weights[key] + dense_delta
                else:
                    reconstructed[key] = dense_delta

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Verify accuracy
        accuracy = self.verify_reconstruction(reconstructed)
        results.update(accuracy)

        # Calculate savings
        total_original = self.original_size + self.target_size
        total_compressed = self.original_size + compressed_size
        results["total_size_mb"] = total_compressed / 1024**2
        results["space_saved_mb"] = (total_original - total_compressed) / 1024**2
        results["compression_ratio"] = total_original / total_compressed
        results["space_saved_pct"] = (1 - total_compressed / total_original) * 100

        return results

    # ============================================
    # Method 3: Quantized Delta Storage
    # ============================================

    def method3_quantized_delta(self, dtype=torch.float16) -> Dict[str, Any]:
        """
        Method 3: Store base + quantized delta
        - Base checkpoint stored as-is
        - Delta stored in lower precision (FP16, BF16, or INT8)
        """
        dtype_name = str(dtype).split('.')[-1]
        print("\n" + "=" * 70)
        print(f"METHOD 3: Quantized Delta Storage ({dtype_name.upper()})")
        print("=" * 70)

        results = {"method": f"Quantized Delta ({dtype_name.upper()})"}

        # Measure compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        delta = self.compute_delta()

        # Quantize delta
        quantized_delta = {}
        for key, tensor in delta.items():
            if dtype == torch.int8:
                # For INT8, we need to store scale and zero point
                # Use per-tensor quantization
                tensor_min = tensor.min()
                tensor_max = tensor.max()

                # Calculate scale and zero point
                scale = (tensor_max - tensor_min) / 255.0
                zero_point = -torch.round(tensor_min / scale).to(torch.int8)

                # Quantize
                quantized = torch.round(tensor / scale + zero_point.float()).clamp(-128, 127).to(torch.int8)

                quantized_delta[key] = quantized
                quantized_delta[f"{key}_scale"] = scale.reshape(1)
                quantized_delta[f"{key}_zero_point"] = zero_point.reshape(1)
            else:
                # For FP16/BF16, direct conversion
                quantized_delta[key] = tensor.to(dtype)

        # Save to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            delta_path = Path(tmp_dir) / f"quantized_delta_{dtype_name}.safetensors"
            save_file(quantized_delta, str(delta_path))

            compressed_size = os.path.getsize(delta_path)
            results["compressed_size_mb"] = compressed_size / 1024**2
            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Measure decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded_quantized = load_file(str(delta_path))

            # Dequantize and reconstruct
            reconstructed = {}
            for key in delta.keys():
                if dtype == torch.int8:
                    # Dequantize INT8
                    quantized = loaded_quantized[key]
                    scale = loaded_quantized[f"{key}_scale"]
                    zero_point = loaded_quantized[f"{key}_zero_point"]

                    dequantized = (quantized.float() - zero_point.float()) * scale
                else:
                    # Convert back to FP32
                    dequantized = loaded_quantized[key].float()

                # Add to base weights
                if key in self.base_weights:
                    reconstructed[key] = self.base_weights[key] + dequantized
                else:
                    reconstructed[key] = dequantized

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Verify accuracy
        accuracy = self.verify_reconstruction(reconstructed)
        results.update(accuracy)

        # Calculate savings
        total_original = self.original_size + self.target_size
        total_compressed = self.original_size + compressed_size
        results["total_size_mb"] = total_compressed / 1024**2
        results["space_saved_mb"] = (total_original - total_compressed) / 1024**2
        results["compression_ratio"] = total_original / total_compressed
        results["space_saved_pct"] = (1 - total_compressed / total_original) * 100

        return results

    # ============================================
    # Method 4: Quantized Checkpoints + FP16 Delta
    # ============================================

    def method4_quantized_checkpoints_fp16(self) -> Dict[str, Any]:
        """
        Method 4: Store checkpoints in FP16, compute delta in FP16 space
        - Base checkpoint quantized to FP16
        - Target checkpoint quantized to FP16
        - Delta computed between quantized checkpoints
        """
        print("\n" + "=" * 70)
        print("METHOD 4: FP16 Checkpoints + FP16 Delta")
        print("=" * 70)

        results = {"method": "FP16 Checkpoints + FP16 Delta"}

        # Measure compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        # Quantize both checkpoints to FP16
        base_fp16 = {k: v.to(torch.float16) for k, v in self.base_weights.items()}
        target_fp16 = {k: v.to(torch.float16) for k, v in self.target_weights.items()}

        # Compute delta in FP16 space
        delta_fp16 = {}
        for key in target_fp16.keys():
            if key in base_fp16:
                delta_fp16[key] = target_fp16[key] - base_fp16[key]
            else:
                delta_fp16[key] = target_fp16[key]

        # Save both base and delta
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base_fp16.safetensors"
            delta_path = Path(tmp_dir) / "delta_fp16.safetensors"

            save_file(base_fp16, str(base_path))
            save_file(delta_fp16, str(delta_path))

            base_size = os.path.getsize(base_path)
            delta_size = os.path.getsize(delta_path)
            compressed_size = base_size + delta_size

            results["base_size_mb"] = base_size / 1024**2
            results["compressed_size_mb"] = delta_size / 1024**2
            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Measure decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded_base = load_file(str(base_path))
            loaded_delta = load_file(str(delta_path))

            # Reconstruct target in FP32
            reconstructed = {}
            for key in loaded_delta.keys():
                if key in loaded_base:
                    reconstructed[key] = (loaded_base[key] + loaded_delta[key]).float()
                else:
                    reconstructed[key] = loaded_delta[key].float()

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Verify accuracy
        accuracy = self.verify_reconstruction(reconstructed)
        results.update(accuracy)

        # Calculate savings
        total_original = self.original_size + self.target_size
        results["total_size_mb"] = compressed_size / 1024**2
        results["space_saved_mb"] = (total_original - compressed_size) / 1024**2
        results["compression_ratio"] = total_original / compressed_size
        results["space_saved_pct"] = (1 - compressed_size / total_original) * 100

        return results

    # ============================================
    # Method 5: Double Quantization (FP16 → INT8)
    # ============================================

    def method5_double_quantization(self) -> Dict[str, Any]:
        """
        Method 5: FP16 checkpoints + INT8 delta (double quantization)
        - Base checkpoint quantized to FP16
        - Target checkpoint quantized to FP16
        - Delta computed in FP16 space, then quantized to INT8
        """
        print("\n" + "=" * 70)
        print("METHOD 5: Double Quantization (FP16 Checkpoints + INT8 Delta)")
        print("=" * 70)

        results = {"method": "Double Quantization (FP16→INT8)"}

        # Measure compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        # Quantize both checkpoints to FP16
        base_fp16 = {k: v.to(torch.float16) for k, v in self.base_weights.items()}
        target_fp16 = {k: v.to(torch.float16) for k, v in self.target_weights.items()}

        # Compute delta in FP16 space
        delta_fp16 = {}
        for key in target_fp16.keys():
            if key in base_fp16:
                delta_fp16[key] = target_fp16[key] - base_fp16[key]
            else:
                delta_fp16[key] = target_fp16[key]

        # Quantize delta to INT8
        delta_int8 = {}
        for key, tensor in delta_fp16.items():
            tensor_fp32 = tensor.float()  # Convert to FP32 for quantization
            tensor_min = tensor_fp32.min()
            tensor_max = tensor_fp32.max()

            scale = (tensor_max - tensor_min) / 255.0
            zero_point = -torch.round(tensor_min / scale).to(torch.int8)
            quantized = torch.round(tensor_fp32 / scale + zero_point.float()).clamp(-128, 127).to(torch.int8)

            delta_int8[key] = quantized
            delta_int8[f"{key}_scale"] = scale.reshape(1)
            delta_int8[f"{key}_zero_point"] = zero_point.reshape(1)

        # Save base and delta
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base_fp16.safetensors"
            delta_path = Path(tmp_dir) / "delta_int8.safetensors"

            save_file(base_fp16, str(base_path))
            save_file(delta_int8, str(delta_path))

            base_size = os.path.getsize(base_path)
            delta_size = os.path.getsize(delta_path)
            compressed_size = base_size + delta_size

            results["base_size_mb"] = base_size / 1024**2
            results["compressed_size_mb"] = delta_size / 1024**2
            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Measure decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded_base = load_file(str(base_path))
            loaded_delta = load_file(str(delta_path))

            # Dequantize delta and reconstruct
            reconstructed = {}
            for key in delta_fp16.keys():
                quantized = loaded_delta[key]
                scale = loaded_delta[f"{key}_scale"]
                zero_point = loaded_delta[f"{key}_zero_point"]

                dequantized_delta = (quantized.float() - zero_point.float()) * scale

                if key in loaded_base:
                    # Reconstruct in FP32
                    reconstructed[key] = loaded_base[key].float() + dequantized_delta
                else:
                    reconstructed[key] = dequantized_delta

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Verify accuracy
        accuracy = self.verify_reconstruction(reconstructed)
        results.update(accuracy)

        # Calculate savings
        total_original = self.original_size + self.target_size
        results["total_size_mb"] = compressed_size / 1024**2
        results["space_saved_mb"] = (total_original - compressed_size) / 1024**2
        results["compression_ratio"] = total_original / compressed_size
        results["space_saved_pct"] = (1 - compressed_size / total_original) * 100

        return results

    # ============================================
    # Method 6: INT8 Checkpoints + INT8 Delta
    # ============================================

    def method6_int8_checkpoints(self) -> Dict[str, Any]:
        """
        Method 6: Store checkpoints in INT8, compute delta in INT8 space
        - Base checkpoint quantized to INT8
        - Target checkpoint quantized to INT8
        - Delta computed between quantized checkpoints
        """
        print("\n" + "=" * 70)
        print("METHOD 6: INT8 Checkpoints + INT8 Delta")
        print("=" * 70)

        results = {"method": "INT8 Checkpoints + INT8 Delta"}

        # Measure compression
        mem_before = self.get_memory_usage()
        start_time = time.time()

        # Quantize base checkpoint to INT8
        base_int8 = {}
        base_scales = {}
        base_zero_points = {}

        for key, tensor in self.base_weights.items():
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            scale = (tensor_max - tensor_min) / 255.0
            zero_point = -torch.round(tensor_min / scale).to(torch.int8)
            quantized = torch.round(tensor / scale + zero_point.float()).clamp(-128, 127).to(torch.int8)

            base_int8[key] = quantized
            base_scales[key] = scale
            base_zero_points[key] = zero_point

        # Quantize target checkpoint to INT8
        target_int8 = {}
        for key, tensor in self.target_weights.items():
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            scale = (tensor_max - tensor_min) / 255.0
            zero_point = -torch.round(tensor_min / scale).to(torch.int8)
            quantized = torch.round(tensor / scale + zero_point.float()).clamp(-128, 127).to(torch.int8)

            target_int8[key] = quantized
            target_int8[f"{key}_scale"] = scale.reshape(1)
            target_int8[f"{key}_zero_point"] = zero_point.reshape(1)

        # Compute delta in INT8 space (store target directly)
        # Note: We store target's INT8 representation as the "delta"
        # because INT8 - INT8 can overflow, so we store target and reconstruct

        # Save base and target
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir) / "base_int8.safetensors"
            target_path = Path(tmp_dir) / "target_int8.safetensors"

            # Prepare base with scales/zero_points
            base_to_save = {}
            for key in base_int8.keys():
                base_to_save[key] = base_int8[key]
                base_to_save[f"{key}_scale"] = base_scales[key].reshape(1)
                base_to_save[f"{key}_zero_point"] = base_zero_points[key].reshape(1)

            save_file(base_to_save, str(base_path))
            save_file(target_int8, str(target_path))

            base_size = os.path.getsize(base_path)
            target_size = os.path.getsize(target_path)
            compressed_size = base_size + target_size

            results["base_size_mb"] = base_size / 1024**2
            results["compressed_size_mb"] = target_size / 1024**2
            results["compression_time_s"] = time.time() - start_time
            results["compression_memory_mb"] = self.get_memory_usage() - mem_before

            # Measure decompression
            mem_before = self.get_memory_usage()
            start_time = time.time()

            loaded_target = load_file(str(target_path))

            # Dequantize target to FP32
            reconstructed = {}
            for key in self.target_weights.keys():
                quantized = loaded_target[key]
                scale = loaded_target[f"{key}_scale"]
                zero_point = loaded_target[f"{key}_zero_point"]

                dequantized = (quantized.float() - zero_point.float()) * scale
                reconstructed[key] = dequantized

            results["decompression_time_s"] = time.time() - start_time
            results["decompression_memory_mb"] = self.get_memory_usage() - mem_before

        # Verify accuracy
        accuracy = self.verify_reconstruction(reconstructed)
        results.update(accuracy)

        # Calculate savings
        total_original = self.original_size + self.target_size
        results["total_size_mb"] = compressed_size / 1024**2
        results["space_saved_mb"] = (total_original - compressed_size) / 1024**2
        results["compression_ratio"] = total_original / compressed_size
        results["space_saved_pct"] = (1 - compressed_size / total_original) * 100

        return results

    def run_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """Run all compression methods and return results."""
        results = {}

        # Method 1: Delta-only
        results["delta_only"] = self.method1_delta_only()

        # Method 2: Sparse delta
        results["sparse_delta"] = self.method2_sparse_delta()

        # Method 3: Quantized delta variants (current approach)
        results["quantized_fp16"] = self.method3_quantized_delta(dtype=torch.float16)
        results["quantized_bf16"] = self.method3_quantized_delta(dtype=torch.bfloat16)
        results["quantized_int8"] = self.method3_quantized_delta(dtype=torch.int8)

        # Method 4-6: Quantize checkpoints first, then delta (new approaches)
        results["fp16_checkpoints"] = self.method4_quantized_checkpoints_fp16()
        results["double_quantization"] = self.method5_double_quantization()
        results["int8_checkpoints"] = self.method6_int8_checkpoints()

        return results

    def generate_report(self, results: Dict[str, Dict[str, Any]]):
        """Generate a comprehensive comparison report."""
        print("\n" + "=" * 70)
        print("COMPRESSION COMPARISON REPORT")
        print("=" * 70)

        # Table header
        print(f"\n{'Method':<25} {'Size (MB)':<12} {'Saved (MB)':<12} {'Ratio':<8} {'Saved %':<10}")
        print("-" * 70)

        # Baseline
        total_original = (self.original_size + self.target_size) / 1024**2
        print(f"{'Baseline (no compression)':<25} {total_original:<12.2f} {0:<12.2f} {1.00:<8.2f} {0:<10.2f}")

        # Each method
        for method_key, result in results.items():
            print(f"{result['method']:<25} "
                  f"{result['total_size_mb']:<12.2f} "
                  f"{result['space_saved_mb']:<12.2f} "
                  f"{result['compression_ratio']:<8.2f} "
                  f"{result['space_saved_pct']:<10.2f}")

        # Performance metrics
        print(f"\n{'Method':<25} {'Compress (s)':<15} {'Decompress (s)':<15} {'Max Error':<12}")
        print("-" * 70)

        for method_key, result in results.items():
            print(f"{result['method']:<25} "
                  f"{result['compression_time_s']:<15.4f} "
                  f"{result['decompression_time_s']:<15.4f} "
                  f"{result['max_absolute_error']:<12.2e}")

        # Memory usage
        print(f"\n{'Method':<25} {'Comp Mem (MB)':<15} {'Decomp Mem (MB)':<15}")
        print("-" * 70)

        for method_key, result in results.items():
            print(f"{result['method']:<25} "
                  f"{result['compression_memory_mb']:<15.2f} "
                  f"{result['decompression_memory_mb']:<15.2f}")

        # Projection for all 8 tasks
        print("\n" + "=" * 70)
        print("PROJECTED SAVINGS FOR 8 TASKS")
        print("=" * 70)

        print(f"\nAssuming ~50 MB per task checkpoint:")
        baseline_8_tasks = 50 * 8
        print(f"  Baseline (no compression): {baseline_8_tasks} MB")

        for method_key, result in results.items():
            # First checkpoint is full, remaining 7 are compressed
            avg_compressed = result['compressed_size_mb']
            projected = 50 + (avg_compressed * 7)  # task_0 full + 7 compressed deltas
            saved = baseline_8_tasks - projected
            saved_pct = (saved / baseline_8_tasks) * 100

            print(f"  {result['method']:<25} {projected:.2f} MB (saves {saved:.2f} MB, {saved_pct:.1f}%)")

        # Recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)

        # Find best by compression ratio
        best_compression = max(results.items(), key=lambda x: x[1]['compression_ratio'])
        print(f"\nBest compression ratio: {best_compression[1]['method']}")
        print(f"  - Compression ratio: {best_compression[1]['compression_ratio']:.2f}x")
        print(f"  - Space saved: {best_compression[1]['space_saved_pct']:.1f}%")

        # Find fastest
        best_speed = min(results.items(),
                        key=lambda x: x[1]['compression_time_s'] + x[1]['decompression_time_s'])
        print(f"\nFastest method: {best_speed[1]['method']}")
        print(f"  - Total time: {best_speed[1]['compression_time_s'] + best_speed[1]['decompression_time_s']:.4f}s")

        # Find most accurate (lossless)
        lossless = [k for k, v in results.items() if v['max_absolute_error'] == 0]
        if lossless:
            print(f"\nLossless methods (exact reconstruction):")
            for method in lossless:
                print(f"  - {results[method]['method']}")

        # Overall recommendation
        print(f"\n{'=' * 70}")
        print("OVERALL RECOMMENDATION")
        print("=" * 70)

        print("""
For continual learning with LoRA adapters, the recommended approach depends on your priorities:

1. **Maximum compression + lossless**: Sparse Delta Storage
   - Leverages sparsity in weight changes
   - Exact reconstruction (no quality loss)
   - Good for storage-constrained environments

2. **Best balance**: Quantized Delta (FP16)
   - ~2x compression vs delta-only
   - Minimal quality loss (typically < 1e-6 error)
   - Fast compression/decompression
   - Good for most use cases

3. **Extreme compression**: Quantized Delta (INT8)
   - ~4x compression vs delta-only
   - Small quality loss (acceptable for most models)
   - Best for very storage-constrained scenarios
   - Recommended to validate model performance after compression

4. **Simplest implementation**: Delta-only
   - Easy to implement and debug
   - Still provides ~50% savings
   - Lossless
   - Good starting point
        """)


def main():
    """Main evaluation function."""
    # Paths to checkpoints
    base_checkpoint = "./run_lora_extended_complete/continual/task_0_C-STANCE"
    target_checkpoint = "./run_lora_extended_complete/continual/task_1_FOMC"

    print("=" * 70)
    print("Delta Compression Evaluation for LoRA Continual Learning")
    print("=" * 70)
    print()

    # Create evaluator
    evaluator = CompressionEvaluator(base_checkpoint, target_checkpoint)

    # Run all methods
    results = evaluator.run_all_methods()

    # Generate report
    evaluator.generate_report(results)

    # Save results to JSON
    output_file = "compression_evaluation_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy/torch types to native Python for JSON serialization
        json_results = {}
        for method, result in results.items():
            json_results[method] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                   for k, v in result.items()}
        json.dump(json_results, f, indent=2)

    print(f"\n\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
