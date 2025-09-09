#!/usr/bin/env python3
"""
Demonstration of what LibTorch does automatically vs manual BLAS
"""
import torch

def show_automatic_dispatch():
    print("=== LibTorch Automatic BLAS Dispatch ===\n")
    
    # Create matrices of different types
    matrices = {
        'float32': torch.randn(512, 512, dtype=torch.float32),
        'float64': torch.randn(512, 512, dtype=torch.float64),
        'complex64': torch.randn(512, 512, dtype=torch.complex64),
        'complex128': torch.randn(512, 512, dtype=torch.complex128),
    }
    
    print("When you call torch.matmul(A, B), LibTorch automatically:")
    print()
    
    for dtype_name, matrix in matrices.items():
        cuda_matrix = matrix.cuda() if torch.cuda.is_available() else matrix
        print(f"• {dtype_name:12} → calls cublas{get_blas_suffix(dtype_name)}gemm")
        
        # Show what happens
        if torch.cuda.is_available():
            result = torch.matmul(cuda_matrix, cuda_matrix)
            print(f"  Result shape: {result.shape}, device: {result.device}")
        print()

def get_blas_suffix(dtype_name):
    """Map PyTorch dtypes to BLAS function suffixes"""
    mapping = {
        'float32': 'S',      # Single precision
        'float64': 'D',      # Double precision  
        'complex64': 'C',    # Complex single
        'complex128': 'Z',   # Complex double
    }
    return mapping.get(dtype_name, 'S')

def show_optimization_features():
    print("=== Additional LibTorch Optimizations ===\n")
    
    optimizations = [
        ("Automatic Algorithm Selection", "Chooses optimal algorithm based on matrix size"),
        ("Memory Layout Optimization", "Handles row-major/column-major automatically"),
        ("Batched Operations", "Processes multiple matrices efficiently"),
        ("Mixed Precision", "Uses Tensor Cores when beneficial"),
        ("Strided Operations", "Works with non-contiguous memory"),
        ("Broadcasting", "Automatic dimension expansion"),
        ("Fusion", "Combines multiple operations"),
        ("Workspace Management", "Optimal temporary memory usage")
    ]
    
    for feature, description in optimizations:
        print(f"✓ {feature:25} - {description}")
    
    print(f"\nAll of this from ONE function call: torch.matmul()")

def show_manual_blas_complexity():
    print("\n=== Manual BLAS Complexity (What You DON'T Want) ===\n")
    
    print("To use raw BLAS, you'd need to handle:")
    complexities = [
        "Choose correct function (sgemm/dgemm/cgemm/zgemm)",
        "Calculate memory strides (lda, ldb, ldc)",
        "Handle matrix transposes (CUBLAS_OP_N/T/C)", 
        "Manage GPU memory manually",
        "Handle different matrix layouts",
        "Choose between regular/batched/strided variants",
        "Set alpha/beta scaling factors",
        "Error checking and cleanup"
    ]
    
    for i, complexity in enumerate(complexities, 1):
        print(f"{i}. {complexity}")
    
    print("\nResult: 50+ lines of code for what torch.matmul() does in 1 line!")

if __name__ == "__main__":
    show_automatic_dispatch()
    show_optimization_features() 
    show_manual_blas_complexity() 