#!/usr/bin/env tclsh

# Load the library
load ./libtorchtcl.so

puts "=== Testing Random Number Generation Operations ==="

# Test manual_seed
puts "\n1. Testing torch::manual_seed"
try {
    set result [torch::manual_seed 42]
    puts "torch::manual_seed 42: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test initial_seed
puts "\n2. Testing torch::initial_seed"
try {
    set result [torch::initial_seed]
    puts "torch::initial_seed: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test seed
puts "\n3. Testing torch::seed"
try {
    set result [torch::seed]
    puts "torch::seed: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test get_rng_state
puts "\n4. Testing torch::get_rng_state"
try {
    set rng_state [torch::get_rng_state]
    puts "torch::get_rng_state: [torch::tensor_shape $rng_state]"
} on error {err} {
    puts "ERROR: $err"
}

# Test set_rng_state
puts "\n5. Testing torch::set_rng_state"
try {
    set state [torch::tensor_create "123" int64]
    set result [torch::set_rng_state $state]
    puts "torch::set_rng_state: $result"
} on error {err} {
    puts "ERROR: $err"
}

# Test bernoulli
puts "\n6. Testing torch::bernoulli"
try {
    set probs [torch::tensor_create "0.5 0.3 0.8" float32]
    set result [torch::bernoulli $probs]
    puts "torch::bernoulli with probs: [torch::tensor_shape $result]"
    
    set result2 [torch::bernoulli $probs 0.7]
    puts "torch::bernoulli with p=0.7: [torch::tensor_shape $result2]"
} on error {err} {
    puts "ERROR: $err"
}

# Test multinomial
puts "\n7. Testing torch::multinomial"
try {
    set weights [torch::tensor_create "1.0 2.0 3.0 4.0" float32]
    set result [torch::multinomial $weights 3 true]
    puts "torch::multinomial: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test normal
puts "\n8. Testing torch::normal"
try {
    set result [torch::normal 0.0 1.0]
    puts "torch::normal single sample: [torch::tensor_shape $result]"
    
    set result2 [torch::normal 0.0 1.0 "2 3"]
    puts "torch::normal shape 2x3: [torch::tensor_shape $result2]"
} on error {err} {
    puts "ERROR: $err"
}

# Test uniform
puts "\n9. Testing torch::uniform"
try {
    set result [torch::uniform "3 3" -1.0 1.0]
    puts "torch::uniform shape 3x3: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test exponential
puts "\n10. Testing torch::exponential"
try {
    set result [torch::exponential "2 2" 1.5]
    puts "torch::exponential shape 2x2: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test gamma
puts "\n11. Testing torch::gamma"
try {
    set result [torch::gamma "2 2" 2.0 1.0]
    puts "torch::gamma shape 2x2: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

# Test poisson
puts "\n12. Testing torch::poisson"
try {
    set result [torch::poisson "2 3" 3.0]
    puts "torch::poisson shape 2x3: [torch::tensor_shape $result]"
} on error {err} {
    puts "ERROR: $err"
}

puts "\n=== Random Number Generation Tests Complete ===" 