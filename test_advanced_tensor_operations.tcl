#!/usr/bin/env tclsh

# Load the library
load ./libtorchtcl.so

puts "=== Testing Advanced Tensor Operations ==="

# Test block_diag
puts "\n1. Testing torch::block_diag"
try {
    set a [torch::tensor_create "1 2 3 4" "2 2" float32]
    set b [torch::tensor_create "5 6 7 8 9" "5 1" float32]
    set result [torch::block_diag $a $b]
    puts "torch::block_diag: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test broadcast_shapes
puts "\n2. Testing torch::broadcast_shapes"
try {
    set shape1 "2 1 3"
    set shape2 "1 4 3"
    set result [torch::broadcast_shapes $shape1 $shape2]
    puts "torch::broadcast_shapes \[2,1,3\] \[1,4,3\]: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test squeeze_multiple
puts "\n3. Testing torch::squeeze_multiple"
try {
    set tensor [torch::tensor_create "1 2 3 4" "1 2 1 2" float32]
    set result [torch::squeeze_multiple $tensor]
    puts "torch::squeeze_multiple (all dims): [torch::tensor_shape $result]"
    
    set result2 [torch::squeeze_multiple $tensor "0 2"]
    puts "torch::squeeze_multiple (dims 0,2): [torch::tensor_shape $result2]"
} on error {err} {
    puts "ERROR: $err"
}

# Test unsqueeze_multiple
puts "\n4. Testing torch::unsqueeze_multiple"
try {
    set tensor [torch::tensor_create "1 2 3 4" "2 2" float32]
    set result [torch::unsqueeze_multiple $tensor "0 2"]
    puts "torch::unsqueeze_multiple: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test tensor_split
puts "\n5. Testing torch::tensor_split"
try {
    set tensor [torch::tensor_create "1 2 3 4 5 6" "6" float32]
    set result [torch::tensor_split $tensor 3]
    puts "torch::tensor_split into 3 sections: [llength $result] tensors"
    
    set result2 [torch::tensor_split $tensor "2 4" 0]
    puts "torch::tensor_split at indices \[2,4\]: [llength $result2] tensors"
} on error {err} {
    puts "ERROR: $err"
}

# Test hsplit
puts "\n6. Testing torch::hsplit"
try {
    set tensor [torch::tensor_create "1 2 3 4 5 6" "2 3" float32]
    set result [torch::hsplit $tensor 3]
    puts "torch::hsplit into 3 sections: [llength $result] tensors"
} on error {err} {
    puts "ERROR: $err"
}

# Test vsplit
puts "\n7. Testing torch::vsplit"
try {
    set tensor [torch::tensor_create "1 2 3 4 5 6" "3 2" float32]
    set result [torch::vsplit $tensor 3]
    puts "torch::vsplit into 3 sections: [llength $result] tensors"
} on error {err} {
    puts "ERROR: Error in vsplit: $err"
}

# Test dsplit
puts "\n8. Testing torch::dsplit"
try {
    set tensor [torch::tensor_create "1 2 3 4 5 6 7 8" "2 2 2" float32]
    set result [torch::dsplit $tensor 2]
    puts "torch::dsplit into 2 sections: [llength $result] tensors"
} on error {err} {
    puts "ERROR: $err"
}

# Test column_stack
puts "\n9. Testing torch::column_stack"
try {
    set a [torch::tensor_create "1 2 3" "3" float32]
    set b [torch::tensor_create "4 5 6" "3" float32]
    set result [torch::column_stack $a $b]
    puts "torch::column_stack: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test row_stack
puts "\n10. Testing torch::row_stack"
try {
    set a [torch::tensor_create "1 2" "1 2" float32]
    set b [torch::tensor_create "3 4" "1 2" float32]
    set result [torch::row_stack $a $b]
    puts "torch::row_stack: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test dstack
puts "\n11. Testing torch::dstack"
try {
    set a [torch::tensor_create "1 2" "2" float32]
    set b [torch::tensor_create "3 4" "2" float32]
    set result [torch::dstack $a $b]
    puts "torch::dstack: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test hstack
puts "\n12. Testing torch::hstack"
try {
    set a [torch::tensor_create "1 2" "2" float32]
    set b [torch::tensor_create "3 4" "2" float32]
    set result [torch::hstack $a $b]
    puts "torch::hstack: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test vstack
puts "\n13. Testing torch::vstack"
try {
    set a [torch::tensor_create "1 2" "1 2" float32]
    set b [torch::tensor_create "3 4" "1 2" float32]
    set result [torch::vstack $a $b]
    puts "torch::vstack: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

puts "\n=== Advanced Tensor Operations Tests Complete ===" 