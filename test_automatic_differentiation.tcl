#!/usr/bin/env tclsh

# Load the library
load ./libtorchtcl.so

puts "=== Testing Automatic Differentiation Operations ==="

# Test grad
puts "\n1. Testing torch::grad"
try {
    set x [torch::tensor_create "1.0 2.0 3.0" float32 "3"]
    set result [torch::grad $x $x]
    puts "torch::grad: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test jacobian
puts "\n2. Testing torch::jacobian"
try {
    set x [torch::tensor_create "1.0 2.0" float32 "2"]
    set result [torch::jacobian $x $x]
    puts "torch::jacobian: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test hessian
puts "\n3. Testing torch::hessian"
try {
    set x [torch::tensor_create "1.0 2.0" float32 "2"]
    set result [torch::hessian $x $x]
    puts "torch::hessian: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test vjp
puts "\n4. Testing torch::vjp"
try {
    set x [torch::tensor_create "1.0 2.0" float32 "2"]
    set v [torch::tensor_create "1.0 0.0" float32 "2"]
    set result [torch::vjp $x $x $v]
    puts "torch::vjp: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test jvp
puts "\n5. Testing torch::jvp"
try {
    set x [torch::tensor_create "1.0 2.0" float32 "2"]
    set v [torch::tensor_create "1.0 0.0" float32 "2"]
    set result [torch::jvp $x $x $v]
    puts "torch::jvp: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test functional_call
puts "\n6. Testing torch::functional_call"
try {
    set func "simple_func"
    set params [torch::tensor_create "1.0 2.0" float32 "2"]
    set inputs [torch::tensor_create "0.5 1.5" float32 "2"]
    set result [torch::functional_call $func $params $inputs]
    puts "torch::functional_call: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test vmap
puts "\n7. Testing torch::vmap"
try {
    set func "batch_func"
    set batch_input [torch::tensor_create "1.0 2.0 3.0 4.0" float32 "2 2"]
    set result [torch::vmap $func $batch_input]
    puts "torch::vmap: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test grad_check
puts "\n8. Testing torch::grad_check"
try {
    set func "test_func"
    set inputs [torch::tensor_create "1.0 2.0" float32 "2"]
    set result [torch::grad_check $func $inputs]
    puts "torch::grad_check: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test grad_check_finite_diff
puts "\n9. Testing torch::grad_check_finite_diff"
try {
    set func "test_func"
    set inputs [torch::tensor_create "1.0 2.0" float32 "2"]
    set eps 1e-6
    set result [torch::grad_check_finite_diff $func $inputs $eps]
    puts "torch::grad_check_finite_diff: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test enable_grad
puts "\n10. Testing torch::enable_grad"
try {
    set result [torch::enable_grad]
    puts "torch::enable_grad: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test no_grad
puts "\n11. Testing torch::no_grad"
try {
    set result [torch::no_grad]
    puts "torch::no_grad: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test set_grad_enabled
puts "\n12. Testing torch::set_grad_enabled"
try {
    set result [torch::set_grad_enabled 1]
    puts "torch::set_grad_enabled true: $result"
    set result [torch::set_grad_enabled 0]
    puts "torch::set_grad_enabled false: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test is_grad_enabled
puts "\n13. Testing torch::is_grad_enabled"
try {
    set result [torch::is_grad_enabled]
    puts "torch::is_grad_enabled: $result"
} on error {err} {
    puts "ERROR: $err"
}

puts "\n=== Automatic Differentiation Operations Testing Complete ===" 