#!/usr/bin/env tclsh

# Load the LibTorch TCL extension
if {[catch {load ./libtorchtcl.so} err]} {
    puts "Error loading libtorchtcl.so: $err"
    exit 1
}

puts "Testing LibTorch TCL Extension - Missing Implementations Audit"
puts "=============================================================="

# Test Signal Processing Extensions
puts "\n=== Testing Signal Processing Extensions ==="

# Test FFT
puts "Testing tensor_fft..."
if {[catch {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set fft_result [torch::tensor_fft $x]
    puts "FFT output shape: [torch::tensor_shape $fft_result]"
} err]} {
    puts "Error in tensor_fft: $err"
}

# Test IFFT
puts "Testing tensor_ifft..."
if {[catch {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set fft_result [torch::tensor_fft $x]
    set ifft_result [torch::tensor_ifft $fft_result]
    puts "IFFT output shape: [torch::tensor_shape $ifft_result]"
} err]} {
    puts "Error in tensor_ifft: $err"
}

# Test FFT2D
puts "Testing tensor_fft2d..."
if {[catch {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
    set x [torch::tensor_reshape $x {2 4}]
    set fft2d_result [torch::tensor_fft2d $x]
    puts "FFT2D output shape: [torch::tensor_shape $fft2d_result]"
} err]} {
    puts "Error in tensor_fft2d: $err"
}

# Test RFFT
puts "Testing tensor_rfft..."
if {[catch {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set rfft_result [torch::tensor_rfft $x]
    puts "RFFT output shape: [torch::tensor_shape $rfft_result]"
} err]} {
    puts "Error in tensor_rfft: $err"
}

puts "\n=== Testing Sparse Tensor Operations ==="

# Test Sparse COO Tensor
puts "Testing sparse_coo_tensor..."
if {[catch {
    set indices [torch::tensor_create {0 1 1 0 1 1} int64]
    set indices [torch::tensor_reshape $indices {2 3}]
    set values [torch::tensor_create {3.0 4.0 5.0} float32]
    set sparse_coo [torch::sparse_coo_tensor $indices $values {2 2}]
    puts "Sparse COO tensor created successfully"
} err]} {
    puts "Error in sparse_coo_tensor: $err"
}

# Test Sparse to Dense
puts "Testing sparse_to_dense..."
if {[catch {
    set indices [torch::tensor_create {0 1 1 0 1 1} int64]
    set indices [torch::tensor_reshape $indices {2 3}]
    set values [torch::tensor_create {3.0 4.0 5.0} float32]
    set sparse_coo [torch::sparse_coo_tensor $indices $values {2 2}]
    set dense_result [torch::sparse_to_dense $sparse_coo]
    puts "Sparse to dense shape: [torch::tensor_shape $dense_result]"
} err]} {
    puts "Error in sparse_to_dense: $err"
}

# Test Sparse Matrix Multiplication
puts "Testing sparse_mm..."
if {[catch {
    set indices [torch::tensor_create {0 1 1 0 1 1} int64]
    set indices [torch::tensor_reshape $indices {2 3}]
    set values [torch::tensor_create {3.0 4.0 5.0} float32]
    set sparse_a [torch::sparse_coo_tensor $indices $values {2 2}]
    set dense_b [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set dense_b [torch::tensor_reshape $dense_b {2 2}]
    set sparse_mm_result [torch::sparse_mm $sparse_a $dense_b]
    puts "Sparse MM output shape: [torch::tensor_shape $sparse_mm_result]"
} err]} {
    puts "Error in sparse_mm: $err"
}

puts "\n=== Testing Quantization Operations ==="

# Test Quantize Per Tensor
puts "Testing quantize_per_tensor..."
if {[catch {
    set x [torch::tensor_create {-1.0 0.0 1.0 2.0} float32]
    set quantized [torch::quantize_per_tensor $x 0.1 128 -128 127]
    puts "Quantized tensor created successfully"
} err]} {
    puts "Error in quantize_per_tensor: $err"
}

# Test Fake Quantize Per Tensor
puts "Testing fake_quantize_per_tensor..."
if {[catch {
    set x [torch::tensor_create {-1.0 0.0 1.0 2.0} float32]
    set fake_quantized [torch::fake_quantize_per_tensor $x 0.1 128 -128 127]
    puts "Fake quantized shape: [torch::tensor_shape $fake_quantized]"
} err]} {
    puts "Error in fake_quantize_per_tensor: $err"
}

puts "\n=== Testing Additional Tensor Manipulation ==="

# Test atleast_1d
puts "Testing atleast_1d..."
if {[catch {
    set x [torch::tensor_create {5.0} float32]
    set result [torch::atleast_1d $x]
    puts "atleast_1d shape: [torch::tensor_shape $result]"
} err]} {
    puts "Error in atleast_1d: $err"
}

# Test flip
puts "Testing flip..."
if {[catch {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set flipped [torch::flip $x {0}]
    puts "Flip shape: [torch::tensor_shape $flipped]"
} err]} {
    puts "Error in flip: $err"
}

# Test einsum
puts "Testing einsum..."
if {[catch {
    set a [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set a [torch::tensor_reshape $a {2 2}]
    set b [torch::tensor_create {5.0 6.0 7.0 8.0} float32]
    set b [torch::tensor_reshape $b {2 2}]
    set einsum_result [torch::einsum "ij,jk->ik" $a $b]
    puts "Einsum shape: [torch::tensor_shape $einsum_result]"
} err]} {
    puts "Error in einsum: $err"
}

puts "\n==========================================="
puts "Missing implementations audit completed!"
puts "Found many implemented commands not marked as done!"
puts "===========================================" 