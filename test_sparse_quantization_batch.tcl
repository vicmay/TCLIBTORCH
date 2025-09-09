#!/usr/bin/env tclsh

# Load the LibTorch extension
if {[catch {load ./libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    exit 1
}

puts "===========================================" 
puts "Testing Sparse Tensor and Quantization Operations"
puts "==========================================="

# Test 1: Basic Sparse COO Tensor
puts "\n--- Test 1: Sparse COO Tensor ---"
if {[catch {
    set indices [torch::tensor_create {0 1 0 1} int64]
    set indices [torch::tensor_reshape $indices {2 2}]
    set values [torch::tensor_create {1.0 2.0} float32]
    set sparse [torch::sparse_coo_tensor $indices $values {2 2}]
    puts "Created sparse COO tensor: $sparse"
    
    set dense [torch::sparse_to_dense $sparse]
    puts "Converted to dense: $dense"
    
    # Note: No cleanup needed - tensors are managed automatically
} err]} {
    puts "Error in Test 1: $err"
}

# Test 2: Quantization Operations
puts "\n--- Test 2: Quantization Operations ---"
if {[catch {
    set input [torch::tensor_randn {3 4}]
    puts "Created input tensor: $input"
    
    # Use fake quantization which works with float tensors  
    set fake_quantized [torch::fake_quantize_per_tensor $input 0.1 0 -128 127]
    puts "Fake quantized tensor: $fake_quantized"
    
    # Note: No cleanup needed - tensors are managed automatically
} err]} {
    puts "Error in Test 2: $err"
}

# Test 3: Sparse Operations
puts "\n--- Test 3: Sparse Matrix Operations ---"
if {[catch {
    set indices [torch::tensor_create {0 1 1 0} int64]
    set indices [torch::tensor_reshape $indices {2 2}]
    set values [torch::tensor_create {3.0 4.0} float32]
    set sparse1 [torch::sparse_coo_tensor $indices $values {2 2}]
    puts "Created sparse1: $sparse1"
    
    set dense_matrix [torch::ones {2 3}]
    puts "Created dense matrix: $dense_matrix"
    
    set result [torch::sparse_mm $sparse1 $dense_matrix]
    puts "Sparse-dense multiplication result: $result"
    
    # Note: No cleanup needed - tensors are managed automatically
} err]} {
    puts "Error in Test 3: $err"
}

# Test 4: Advanced Sparse Operations
puts "\n--- Test 4: Advanced Sparse Operations ---"
if {[catch {
    set indices [torch::tensor_create {0 1 1 0 0 1} int64]
    set indices [torch::tensor_reshape $indices {2 3}]
    set values [torch::tensor_create {1.0 2.0 3.0} float32]
    set sparse [torch::sparse_coo_tensor $indices $values {2 2}]
    puts "Created sparse with duplicates: $sparse"
    
    set coalesced [torch::sparse_coalesce $sparse]
    puts "Coalesced sparse tensor: $coalesced"
    
    # Note: No cleanup needed - tensors are managed automatically
} err]} {
    puts "Error in Test 4: $err"
}

# Test 5: Fake Quantization Properties  
puts "\n--- Test 5: Fake Quantization Properties ---"
if {[catch {
    set input [torch::tensor_randn {2 3}]
    puts "Created input tensor: $input"
    
    # Use fake quantization which simulates quantization without changing tensor type
    set fake_quantized [torch::fake_quantize_per_tensor $input 0.1 0 -128 127]
    puts "Fake quantized tensor: $fake_quantized"
    
    # Note: No cleanup needed - tensors are managed automatically
} err]} {
    puts "Error in Test 5: $err"
}

puts "\n==========================================="
puts "Sparse tensor and quantization operations test completed!"
puts "===========================================" 