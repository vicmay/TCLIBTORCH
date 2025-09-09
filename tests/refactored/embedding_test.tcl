#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the shared library
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test parameters
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper procedure to create test tensors
proc create_tensor {data dims {dtype int64}} {
    return [torch::tensor_create -data $data -dtype $dtype -shape $dims]
}

# Helper procedure to check tensor values approximately
proc tensor_approx_equal {t1 t2 {tolerance 1e-6}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# ========================================
# Tests for Positional Syntax (Backward Compatibility)
# ========================================

test embedding-1.1 {Basic embedding lookup - positional syntax} {
    # Create indices tensor [0, 1, 2]
    set indices [create_tensor {0 1 2} {3} int64]
    set result [torch::embedding $indices 5 4]
    
    # Check output shape should be [3, 4] (3 indices, 4-dim embeddings)
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 4"}
} 1

test embedding-1.2 {Embedding with padding - positional syntax} {
    # Create indices with padding index
    set indices [create_tensor {0 1 3 2} {4} int64]
    # Use padding_idx = 1
    set result [torch::embedding $indices 5 3 1]
    
    # Check output shape should be [4, 3]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "4 3"}
} 1

test embedding-1.3 {2D indices tensor - positional syntax} {
    # Create 2D indices tensor
    set indices [create_tensor {0 1 2 3} {2 2} int64]
    set result [torch::embedding $indices 6 8]
    
    # Check output shape should be [2, 2, 8]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2 8"}
} 1

test embedding-1.4 {Large vocabulary - positional syntax} {
    # Test with larger vocabulary
    set indices [create_tensor {0 99 50} {3} int64]
    set result [torch::embedding $indices 100 16]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 16"}
} 1

test embedding-1.5 {Single index lookup - positional syntax} {
    # Single index
    set indices [create_tensor {7} {1} int64]
    set result [torch::embedding $indices 10 5]
    
    # Check output shape should be [1, 5]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 5"}
} 1

# ========================================
# Tests for Named Parameter Syntax
# ========================================

test embedding-2.1 {Basic embedding - named syntax with -input} {
    set indices [create_tensor {0 1 2} {3} int64]
    set result [torch::embedding -input $indices -num_embeddings 5 -embedding_dim 4]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 4"}
} 1

test embedding-2.2 {Named syntax with -tensor parameter} {
    set indices [create_tensor {0 1 2} {3} int64]
    set result [torch::embedding -tensor $indices -num_embeddings 5 -embedding_dim 4]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 4"}
} 1

test embedding-2.3 {Named syntax with padding_idx} {
    set indices [create_tensor {0 1 3 2} {4} int64]
    set result [torch::embedding -input $indices -num_embeddings 5 -embedding_dim 3 -padding_idx 1]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "4 3"}
} 1

test embedding-2.4 {Named syntax with all parameters} {
    set indices [create_tensor {0 2 4} {3} int64]
    set result [torch::embedding -input $indices -num_embeddings 6 -embedding_dim 8 -padding_idx 0]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 8"}
} 1

test embedding-2.5 {Parameter order independence} {
    set indices [create_tensor {1 2 3} {3} int64]
    set result [torch::embedding -embedding_dim 4 -input $indices -num_embeddings 5]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 4"}
} 1

# ========================================
# Tests for CamelCase Alias
# ========================================

test embedding-3.1 {CamelCase alias - positional syntax} {
    set indices [create_tensor {0 1 2} {3} int64]
    set result [torch::Embedding $indices 5 4]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 4"}
} 1

test embedding-3.2 {CamelCase alias - named syntax} {
    set indices [create_tensor {0 1 2} {3} int64]
    set result [torch::Embedding -input $indices -num_embeddings 5 -embedding_dim 4]
    
    # Check output shape
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 4"}
} 1

# ========================================
# Tests for Syntax Consistency
# ========================================

test embedding-4.1 {Syntax consistency - same result structure} {
    set indices [create_tensor {0 1 2} {3} int64]
    
    set result1 [torch::embedding $indices 5 4]
    set result2 [torch::embedding -input $indices -num_embeddings 5 -embedding_dim 4]
    
    # Check both have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

test embedding-4.2 {CamelCase produces same shape} {
    set indices [create_tensor {0 1 2} {3} int64]
    
    set result1 [torch::embedding $indices 5 4]
    set result2 [torch::Embedding $indices 5 4]
    
    # Check both have same shape
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} 1

# ========================================
# Tests for Different Data Types and Shapes
# ========================================

test embedding-5.1 {Different embedding dimensions} {
    set indices [create_tensor {0 1} {2} int64]
    
    # Test different embedding dimensions
    set result_small [torch::embedding $indices 3 2]
    set result_large [torch::embedding $indices 3 100]
    
    set shape_small [torch::tensor_shape $result_small]
    set shape_large [torch::tensor_shape $result_large]
    
    expr {$shape_small eq "2 2" && $shape_large eq "2 100"}
} 1

test embedding-5.2 {Different vocabulary sizes} {
    set indices [create_tensor {0 1 2} {3} int64]
    
    # Test different vocabulary sizes
    set result_small [torch::embedding $indices 5 4]
    set result_large [torch::embedding $indices 1000 4]
    
    set shape_small [torch::tensor_shape $result_small]
    set shape_large [torch::tensor_shape $result_large]
    
    expr {$shape_small eq "3 4" && $shape_large eq "3 4"}
} 1

test embedding-5.3 {3D input tensor} {
    # Create 3D indices tensor (batch, sequence)
    set indices [create_tensor {0 1 2 1 0 3} {2 3} int64]
    set result [torch::embedding $indices 5 8]
    
    # Check output shape should be [2, 3, 8]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 3 8"}
} 1

test embedding-5.4 {Different padding indices} {
    set indices [create_tensor {0 1 2 3} {4} int64]
    
    # Test with different padding indices
    set result_pad0 [torch::embedding $indices 5 4 0]
    set result_pad2 [torch::embedding $indices 5 4 2]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $result_pad0]
    set shape2 [torch::tensor_shape $result_pad2]
    expr {$shape1 eq $shape2 && $shape1 eq "4 4"}
} 1

# ========================================
# Tests for Edge Cases
# ========================================

test embedding-6.1 {Minimum vocabulary size} {
    set indices [create_tensor {0} {1} int64]
    set result [torch::embedding $indices 1 1]
    
    set shape [torch::tensor_shape $result]
    expr {$shape eq "1 1"}
} 1

test embedding-6.2 {Large indices values} {
    # Use indices near vocabulary boundary
    set indices [create_tensor {0 49 99} {3} int64]
    set result [torch::embedding $indices 100 4]
    
    set shape [torch::tensor_shape $result]
    expr {$shape eq "3 4"}
} 1

test embedding-6.3 {Sequential indices} {
    # Test with sequential indices
    set indices [create_tensor {0 1 2 3 4} {5} int64]
    set result [torch::embedding $indices 6 3]
    
    set shape [torch::tensor_shape $result]
    expr {$shape eq "5 3"}
} 1

# ========================================
# Error Handling Tests
# ========================================

test embedding-7.1 {Error - missing arguments} {
    set result [catch {torch::embedding} error]
    set result
} 1

test embedding-7.2 {Error - insufficient positional arguments} {
    set indices [create_tensor {0 1} {2} int64]
    set result [catch {torch::embedding $indices} error]
    set result
} 1

test embedding-7.3 {Error - invalid tensor name} {
    set result [catch {torch::embedding "invalid_tensor" 5 4} error]
    set result
} 1

test embedding-7.4 {Error - zero num_embeddings} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding $indices 0 4} error]
    set result
} 1

test embedding-7.5 {Error - negative num_embeddings} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding $indices -5 4} error]
    set result
} 1

test embedding-7.6 {Error - zero embedding_dim} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding $indices 5 0} error]
    set result
} 1

test embedding-7.7 {Error - negative embedding_dim} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding $indices 5 -4} error]
    set result
} 1

test embedding-7.8 {Error - missing value for named parameter} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding -input $indices -num_embeddings} error]
    set result
} 1

test embedding-7.9 {Error - unknown named parameter} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding -input $indices -unknown_param 5} error]
    set result
} 1

test embedding-7.10 {Error - invalid num_embeddings in named syntax} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding -input $indices -num_embeddings "not_a_number" -embedding_dim 4} error]
    set result
} 1

test embedding-7.11 {Error - invalid embedding_dim in named syntax} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding -input $indices -num_embeddings 5 -embedding_dim "not_a_number"} error]
    set result
} 1

test embedding-7.12 {Error - invalid padding_idx} {
    set indices [create_tensor {0} {1} int64]
    set result [catch {torch::embedding $indices 5 4 "not_a_number"} error]
    set result
} 1

test embedding-7.13 {Error - CamelCase with invalid parameters} {
    set result [catch {torch::Embedding "invalid_tensor" 5 4} error]
    set result
} 1

# ========================================
# Integration Tests
# ========================================

test embedding-8.1 {Integration with tensor operations} {
    # Create embedding and perform operations on result
    set indices [create_tensor {0 1 2} {3} int64]
    set embeddings [torch::embedding $indices 5 4]
    
    # Sum over embedding dimension
    set summed [torch::tensor_sum $embeddings 1]
    
    # Should result in shape [3] (one sum per index)
    set shape [torch::tensor_shape $summed]
    expr {$shape eq "3"}
} 1

test embedding-8.2 {Multiple embedding operations} {
    # Test creating multiple embeddings
    set indices1 [create_tensor {0 1} {2} int64]
    set indices2 [create_tensor {2 3} {2} int64]
    
    set emb1 [torch::embedding $indices1 5 3]
    set emb2 [torch::embedding $indices2 5 3]
    
    # Both should have same shape
    set shape1 [torch::tensor_shape $emb1]
    set shape2 [torch::tensor_shape $emb2]
    expr {$shape1 eq $shape2 && $shape1 eq "2 3"}
} 1

test embedding-8.3 {Embedding with downstream processing} {
    # Create sentence indices
    set sentence [create_tensor {1 5 3 2 4} {5} int64]
    set embeddings [torch::embedding $sentence 10 8]
    
    # Mean pooling over sequence dimension
    set pooled [torch::tensor_mean $embeddings 0]
    
    # Should result in shape [8] (embedding dimension)
    set shape [torch::tensor_shape $pooled]
    expr {$shape eq "8"}
} 1

# Clean up any remaining tensors and run tests
cleanupTests 