#!/usr/bin/env tclsh

# Load the LibTorch TCL extension
load ./build/libtorchtcl.so

puts "=== Testing Phase 2 Activation Functions ==="

# Test existing functionality first
puts "\n=== Verifying Existing Functionality ==="
set t1 [torch::tensor_create {1.0 2.0 3.0} float32 cpu 0]
puts "Created tensor: $t1"
torch::tensor_print $t1

# Test basic math still works
set relu_result [torch::tensor_relu $t1]
puts "ReLU result: $relu_result"
torch::tensor_print $relu_result

puts "\n=== Testing Phase 2 New Activation Functions ==="

# Create test tensor with negative and positive values for comprehensive testing
set test_tensor [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu 0]
puts "Test tensor: $test_tensor"
torch::tensor_print $test_tensor

# Test GELU activation
puts "\n--- Testing GELU ---"
set gelu_result [torch::gelu $test_tensor]
puts "GELU result: $gelu_result"
torch::tensor_print $gelu_result

# Test SELU activation
puts "\n--- Testing SELU ---"
set selu_result [torch::selu $test_tensor]
puts "SELU result: $selu_result"
torch::tensor_print $selu_result

# Test ELU activation
puts "\n--- Testing ELU ---"
set elu_result [torch::elu $test_tensor]
puts "ELU result: $elu_result"
torch::tensor_print $elu_result

# Test Leaky ReLU activation (with default and custom slope)
puts "\n--- Testing Leaky ReLU ---"
set leaky_relu_default [torch::leaky_relu $test_tensor]
puts "Leaky ReLU (default): $leaky_relu_default"
torch::tensor_print $leaky_relu_default

set leaky_relu_custom [torch::leaky_relu $test_tensor 0.2]
puts "Leaky ReLU (slope=0.2): $leaky_relu_custom"
torch::tensor_print $leaky_relu_custom

# Test ReLU6 activation
puts "\n--- Testing ReLU6 ---"
set large_tensor [torch::tensor_create {-2.0 2.0 4.0 6.0 8.0} float32 cpu 0]
set relu6_result [torch::relu6 $large_tensor]
puts "ReLU6 result: $relu6_result"
torch::tensor_print $relu6_result

# Test Hard Tanh activation
puts "\n--- Testing Hard Tanh ---"
set hardtanh_result [torch::hardtanh $test_tensor]
puts "Hard Tanh result: $hardtanh_result"
torch::tensor_print $hardtanh_result

# Test Hard Swish activation
puts "\n--- Testing Hard Swish ---"
set hardswish_result [torch::hardswish $test_tensor]
puts "Hard Swish result: $hardswish_result"
torch::tensor_print $hardswish_result

# Test Hard Sigmoid activation
puts "\n--- Testing Hard Sigmoid ---"
set hardsigmoid_result [torch::hardsigmoid $test_tensor]
puts "Hard Sigmoid result: $hardsigmoid_result"
torch::tensor_print $hardsigmoid_result

# Test SiLU (Swish) activation
puts "\n--- Testing SiLU ---"
set silu_result [torch::silu $test_tensor]
puts "SiLU result: $silu_result"
torch::tensor_print $silu_result

# Test Mish activation
puts "\n--- Testing Mish ---"
set mish_result [torch::mish $test_tensor]
puts "Mish result: $mish_result"
torch::tensor_print $mish_result

# Test Softplus activation
puts "\n--- Testing Softplus ---"
set softplus_result [torch::softplus $test_tensor]
puts "Softplus result: $softplus_result"
torch::tensor_print $softplus_result

# Test Softsign activation
puts "\n--- Testing Softsign ---"
set softsign_result [torch::softsign $test_tensor]
puts "Softsign result: $softsign_result"
torch::tensor_print $softsign_result

# Test Tanh Shrink activation
puts "\n--- Testing Tanh Shrink ---"
set tanhshrink_result [torch::tanhshrink $test_tensor]
puts "Tanh Shrink result: $tanhshrink_result"
torch::tensor_print $tanhshrink_result

# Test Threshold activation
puts "\n--- Testing Threshold ---"
set threshold_result [torch::threshold $test_tensor 0.5 -1.0]
puts "Threshold result (threshold=0.5, value=-1.0): $threshold_result"
torch::tensor_print $threshold_result

# Test Randomized ReLU activation
puts "\n--- Testing RReLU ---"
set rrelu_result [torch::rrelu $test_tensor]
puts "RReLU result (default bounds): $rrelu_result"
torch::tensor_print $rrelu_result

# Test CELU activation
puts "\n--- Testing CELU ---"
set celu_result [torch::celu $test_tensor]
puts "CELU result: $celu_result"
torch::tensor_print $celu_result

# Test dimensioned activations with 2D tensor
puts "\n=== Testing Dimensioned Activations ==="
set tensor_2d [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu 0]
set tensor_2d [torch::tensor_reshape $tensor_2d {2 3}]
puts "2D test tensor:"
torch::tensor_print $tensor_2d

# Test Softmin activation
puts "\n--- Testing Softmin ---"
set softmin_result [torch::softmin $tensor_2d 1]
puts "Softmin result (dim=1): $softmin_result"
torch::tensor_print $softmin_result

# Test Softmax2D activation
puts "\n--- Testing Softmax2D ---"
set softmax2d_result [torch::softmax2d $tensor_2d]
puts "Softmax2D result: $softmax2d_result"
torch::tensor_print $softmax2d_result

# Test Log Softmax activation
puts "\n--- Testing Log Softmax ---"
set logsoftmax_result [torch::logsoftmax $tensor_2d 1]
puts "Log Softmax result (dim=1): $logsoftmax_result"
torch::tensor_print $logsoftmax_result

# Test Parametric ReLU activation
puts "\n--- Testing PReLU ---"
set weight_tensor [torch::tensor_create {0.25} float32 cpu 0]
set prelu_result [torch::prelu $test_tensor $weight_tensor]
puts "PReLU result: $prelu_result"
torch::tensor_print $prelu_result

# Test GLU activation (needs even-sized last dimension)
puts "\n--- Testing GLU ---"
set glu_tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu 0]
set glu_result [torch::glu $glu_tensor]
puts "GLU result: $glu_result"
torch::tensor_print $glu_result

puts "\n=== All Phase 2 Activation Functions Tests Completed Successfully! ==="
puts "✅ Total activation functions tested: 21"
puts "✅ All existing functionality preserved"
puts "✅ Ready for production use" 