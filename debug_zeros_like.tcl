#!/usr/bin/env tclsh

# Load extension
if {[catch {load ./build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Create test tensor
puts "Creating test tensor..."
set test_tensor [torch::ones {3 2}]
puts "Test tensor: $test_tensor"

# Test basic zeros_like
puts "\nTest 1: Basic zeros_like"
set result1 [torch::zeros_like $test_tensor]
set shape1 [torch::tensor_shape $result1]
puts "Shape: $shape1"

# Test zeros_like with dtype
puts "\nTest 2: zeros_like with dtype"
set result2 [torch::zeros_like $test_tensor int32]
set dtype2 [torch::tensor_dtype $result2]
puts "Dtype: $dtype2"
puts "Comparison: [expr {$dtype2 eq "Int32"}]"

# Test named parameter syntax
puts "\nTest 3: Named parameter syntax"
set result3 [torch::zeros_like -input $test_tensor -dtype int64]
set dtype3 [torch::tensor_dtype $result3]
puts "Dtype: $dtype3"
puts "Comparison: [expr {$dtype3 eq "Int64"}]"

# Test camelCase alias
puts "\nTest 4: CamelCase alias"
set result4 [torch::zerosLike -input $test_tensor -dtype float64]
set dtype4 [torch::tensor_dtype $result4]
puts "Dtype: $dtype4"
puts "Comparison: [expr {$dtype4 eq "Float64"}]"

puts "\nDone."
