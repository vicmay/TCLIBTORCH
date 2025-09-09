#!/usr/bin/env tclsh

if {[catch {load ../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

# Create tensors with different dtypes
set tensor1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
puts "Tensor1 dtype: [torch::tensor_dtype $tensor1]"

set tensor2 [torch::tensor_create -data {1.0 2.0} -dtype float64]
puts "Tensor2 dtype: [torch::tensor_dtype $tensor2]"

set tensor3 [torch::tensor_create -data {1 2} -dtype int32]
puts "Tensor3 dtype: [torch::tensor_dtype $tensor3]"

set tensor4 [torch::tensor_create -data {1 2} -dtype int64]
puts "Tensor4 dtype: [torch::tensor_dtype $tensor4]"
