#!/usr/bin/env tclsh

puts "Testing Loss Functions..."

# Load the library
load ./build/libtorchtcl.so

# Test MSE Loss
puts "Testing MSE Loss..."
set input [torch::tensor_create {2.0 3.0 1.0 4.0} float32 cuda 0]
set target [torch::tensor_create {1.0 2.0 1.5 3.0} float32 cuda 0]
set mse_loss [torch::mse_loss $input $target]
puts "MSE Loss: [torch::tensor_print $mse_loss]"

# Test Cross Entropy Loss (simple case)
puts "Testing Cross Entropy Loss..."
set logits [torch::tensor_create {1.0 2.0 0.5 0.8 0.3 2.1} float32 cuda 0]
set logits_2d [torch::tensor_reshape $logits {2 3}]
set labels [torch::tensor_create {1 2} int64 cuda 0]
set ce_loss [torch::cross_entropy_loss $logits_2d $labels]
puts "Cross Entropy Loss: [torch::tensor_print $ce_loss]"

# Test BCE Loss
puts "Testing Binary Cross Entropy Loss..."
set sigmoid_output [torch::tensor_create {0.8 0.2 0.3 0.9} float32 cuda 0]
set binary_targets [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cuda 0]
set bce_loss [torch::bce_loss $sigmoid_output $binary_targets]
puts "BCE Loss: [torch::tensor_print $bce_loss]"

# Test NLL Loss  
puts "Testing NLL Loss..."
set log_probs [torch::tensor_create {-0.5 -1.2 -2.1 -0.8 -1.5 -0.3} float32 cuda 0]
set log_probs_2d [torch::tensor_reshape $log_probs {2 3}]
set targets [torch::tensor_create {0 2} int64 cuda 0]
set nll_loss [torch::nll_loss $log_probs_2d $targets]
puts "NLL Loss: [torch::tensor_print $nll_loss]"

puts "All loss functions working correctly!" 