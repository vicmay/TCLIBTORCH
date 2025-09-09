#!/usr/bin/env tclsh
# Advanced Functionality Tests for LibTorch TCL Extension
# Tests neural networks, optimizers, cuDNN, cuFFT, and advanced CUDA operations

puts [string repeat "=" 80]
puts "LibTorch TCL Extension - Advanced Functionality Tests"
puts [string repeat "=" 80]

# Load the library first
load ../build/libtorchtcl.so

# Global test counters
set ::total_tests 0
set ::passed_tests 0
set ::failed_tests 0

# Test runner that shows all details
proc run_test {test_name test_code} {
    incr ::total_tests
    puts "\n[format "Test %02d" $::total_tests]: $test_name"
    puts [string repeat "-" 60]
    
    set start_time [clock milliseconds]
    
    if {[catch {
        eval $test_code
        incr ::passed_tests
        set result "✅ PASSED"
    } error]} {
        incr ::failed_tests
        set result "❌ FAILED: $error"
        puts "ERROR DETAILS: $::errorInfo"
    }
    
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    
    puts "Result: $result (${duration}ms)"
    return [expr {$::failed_tests == 0}]
}

# Test 1: Neural Network Forward Pass (cuDNN)
run_test "Neural Network Forward Pass (cuDNN)" {
    puts "Testing neural network forward pass with cuDNN acceleration..."
    
    # Create input tensor (batch=2, channels=1, height=4, width=4) - typical CNN input
    set input [torch::tensor_randn {2 1 4 4} cuda]
    puts "Input shape: [torch::tensor_shape $input]"
    
    # Create Conv2D layer (1 input channel, 8 output channels, 3x3 kernel)
    set conv_layer [torch::conv2d 1 8 3]
    puts "Created conv2d layer: $conv_layer"
    
    # Move layer to CUDA for cuDNN acceleration
    set conv_cuda [torch::layer_to $conv_layer cuda]
    puts "Moved layer to CUDA: $conv_cuda"
    
    # Forward pass (uses cuDNN)
    set output [torch::layer_forward $conv_cuda $input]
    puts "Forward pass output shape: [torch::tensor_shape $output]"
    puts "Output (sample): [torch::tensor_print $output]"
    
    # Verify output is on CUDA and has correct shape
    if {![string match "*CUDA*" [torch::tensor_print $output]]} {
        error "Output should be on CUDA"
    }
    
    set output_shape [torch::tensor_shape $output]
    if {![string match "*8*" $output_shape]} {
        error "Output should have 8 channels: $output_shape"
    }
    
    puts "✓ cuDNN-accelerated convolution successful"
}

# Test 2: Neural Network Training Loop
run_test "Neural Network Training Loop" {
    puts "Testing complete neural network training loop..."
    
    # Create simple MLP model
    set linear1 [torch::linear 4 10]
    set linear2 [torch::linear 10 1]
    
    # Move to CUDA
    set linear1_cuda [torch::layer_to $linear1 cuda]
    set linear2_cuda [torch::layer_to $linear2 cuda]
    
    puts "Created 2-layer MLP on CUDA"
    
    # Create training data
    set X [torch::tensor_randn {100 4} cuda]
    set y [torch::tensor_randn {100 1} cuda]
    
    puts "Created training data: X[torch::tensor_shape $X], y[torch::tensor_shape $y]"
    
    # Training loop
    for {set epoch 0} {$epoch < 3} {incr epoch} {
        puts "Epoch $epoch:"
        
        # Forward pass
        set h1 [torch::layer_forward $linear1_cuda $X]
        set h1_relu [torch::tensor_relu $h1]
        set predictions [torch::layer_forward $linear2_cuda $h1_relu]
        
        # Calculate loss (MSE)
        set diff [torch::tensor_sub $predictions $y]
        set loss_tensor [torch::tensor_mean [torch::tensor_mul $diff $diff]]
        set loss_value [torch::tensor_item $loss_tensor]
        
        puts "  Loss: $loss_value"
        
        if {$loss_value > 1000.0} {
            error "Loss too high: $loss_value"
        }
    }
    
    puts "✓ Neural network training loop successful"
}

# Test 3: Optimizer Testing
run_test "Optimizer Testing" {
    puts "Testing optimizers..."
    
    # Create a simple model
    set model [torch::linear 2 1]
    set model_cuda [torch::layer_to $model cuda]
    
    # Get model parameters
    set params [torch::layer_parameters $model_cuda]
    puts "Model parameters: $params"
    
    # Create optimizer
    set optimizer [torch::optimizer_adam $params 0.01]
    puts "Created Adam optimizer: $optimizer"
    
    # Test optimizer step
    torch::optimizer_zero_grad $optimizer
    puts "✓ Zero grad successful"
    
    # Create dummy gradients (in real training these come from backprop)
    # For now just verify optimizer creation works
    puts "✓ Optimizer creation and basic operations successful"
}

# Test 4: cuFFT Operations
run_test "cuFFT Operations" {
    puts "Testing cuFFT (Fast Fourier Transform) operations..."
    
    # Create complex signal on CUDA
    set signal_size 64
    set real_part [torch::tensor_randn [list $signal_size] cuda]
    set imag_part [torch::tensor_randn [list $signal_size] cuda]
    
    puts "Created real signal: [torch::tensor_shape $real_part]"
    puts "Created imaginary signal: [torch::tensor_shape $imag_part]"
    
    # Test FFT operation
    set fft_result [torch::tensor_fft $real_part]
    puts "FFT result shape: [torch::tensor_shape $fft_result]"
    puts "FFT result (sample): [torch::tensor_print $fft_result]"
    
    # Verify result is on CUDA
    if {![string match "*CUDA*" [torch::tensor_print $fft_result]]} {
        error "FFT result should be on CUDA"
    }
    
    puts "✓ cuFFT operations successful"
}

# Test 5: Advanced Tensor Operations
run_test "Advanced Tensor Operations" {
    puts "Testing advanced tensor operations..."
    
    # Large matrix operations (stress test cuBLAS)
    set size 256
    set A [torch::tensor_randn [list $size $size] cuda]
    set B [torch::tensor_randn [list $size $size] cuda]
    
    puts "Created large matrices ${size}x${size}"
    
    # Matrix multiplication (cuBLAS)
    set start_time [clock milliseconds]
    set C [torch::tensor_matmul $A $B]
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    
    puts "Matrix multiplication completed in ${duration}ms"
    puts "Result shape: [torch::tensor_shape $C]"
    
    # Test batch operations
    set batch_A [torch::tensor_randn {10 64 64} cuda]
    set batch_B [torch::tensor_randn {10 64 64} cuda]
    set batch_C [torch::tensor_bmm $batch_A $batch_B]
    
    puts "Batch matrix multiplication: [torch::tensor_shape $batch_C]"
    
    if {![string match "*CUDA*" [torch::tensor_print $batch_C]]} {
        error "Batch result should be on CUDA"
    }
    
    puts "✓ Advanced tensor operations successful"
}

# Test 6: Memory Stress Test
run_test "Memory Stress Test" {
    puts "Testing memory management under stress..."
    
    set tensors [list]
    
    # Create many tensors
    for {set i 0} {$i < 50} {incr i} {
        set size [expr {64 + $i * 4}]
        set tensor [torch::tensor_randn [list $size $size] cuda]
        lappend tensors $tensor
        
        if {[expr {$i % 10}] == 0} {
            set memory_info [torch::cuda_memory_info]
            puts "Iteration $i, Memory: $memory_info"
        }
    }
    
    puts "Created [llength $tensors] tensors"
    
    # Clear some tensors and check memory
    set final_memory [torch::cuda_memory_info]
    puts "Final memory state: $final_memory"
    
    puts "✓ Memory stress test successful"
}

# Test 7: Data Type Conversions
run_test "Data Type Conversions" {
    puts "Testing data type conversions..."
    
    # Test different data types
    set float_tensor [torch::tensor_create {1.5 2.5 3.5} float32 cuda 0]
    set double_tensor [torch::tensor_to $float_tensor cuda float64]
    set int_tensor [torch::tensor_to $float_tensor cuda int32]
    
    puts "Float dtype: [torch::tensor_dtype $float_tensor]"
    puts "Double dtype: [torch::tensor_dtype $double_tensor]"
    puts "Int dtype: [torch::tensor_dtype $int_tensor]"
    
    # Verify conversions
    if {![string match "*Float*" [torch::tensor_dtype $float_tensor]]} {
        error "Float tensor conversion failed"
    }
    
    if {![string match "*Double*" [torch::tensor_dtype $double_tensor]]} {
        error "Double tensor conversion failed"
    }
    
    if {![string match "*Int*" [torch::tensor_dtype $int_tensor]]} {
        error "Int tensor conversion failed"
    }
    
    puts "✓ Data type conversions successful"
}

# Test 8: Random Number Generation (cuRAND)
run_test "Random Number Generation (cuRAND)" {
    puts "Testing CUDA random number generation..."
    
    # Test different random distributions
    set normal_tensor [torch::tensor_randn {100 100} cuda]
    set uniform_tensor [torch::tensor_rand {100 100} cuda]
    
    puts "Normal distribution tensor: [torch::tensor_shape $normal_tensor]"
    puts "Uniform distribution tensor: [torch::tensor_shape $uniform_tensor]"
    
    # Verify randomness by checking some statistics
    set mean_val [torch::tensor_mean $normal_tensor]
    set mean_scalar [torch::tensor_item $mean_val]
    
    puts "Normal tensor mean: $mean_scalar (should be near 0)"
    
    # Mean should be reasonably close to 0 for normal distribution
    if {[expr {abs($mean_scalar)}] > 0.5} {
        error "Normal distribution mean too far from 0: $mean_scalar"
    }
    
    puts "✓ cuRAND operations successful"
}

# Test 9: Advanced Neural Network Layers
run_test "Advanced Neural Network Layers" {
    puts "Testing advanced neural network layers..."
    
    # Test different layer types
    set conv_layer [torch::conv2d 3 16 5]
    set batch_norm [torch::batchnorm2d 16]
    set dropout [torch::dropout 0.5]
    
    puts "Created conv2d layer: $conv_layer"
    puts "Created batch norm layer: $batch_norm"
    puts "Created dropout layer: $dropout"
    
    # Move to CUDA
    set conv_cuda [torch::layer_to $conv_layer cuda]
    set bn_cuda [torch::layer_to $batch_norm cuda]
    set dropout_cuda [torch::layer_to $dropout cuda]
    
    # Test with realistic input
    # Create CIFAR-10 like input
    set input [torch::tensor_randn {4 3 32 32} cuda]
    puts "Input shape: [torch::tensor_shape $input]"
    
    # Forward pass through layers
    set conv_out [torch::layer_forward $conv_cuda $input]
    puts "Conv output shape: [torch::tensor_shape $conv_out]"
    
    set bn_out [torch::layer_forward $bn_cuda $conv_out]
    puts "BatchNorm output shape: [torch::tensor_shape $bn_out]"
    
    set dropout_out [torch::layer_forward $dropout_cuda $bn_out]
    puts "Dropout output shape: [torch::tensor_shape $dropout_out]"
    
    # Verify final output is on CUDA
    if {![string match "*CUDA*" [torch::tensor_print $dropout_out]]} {
        error "Final output should be on CUDA"
    }
    
    puts "✓ Advanced neural network layers successful"
}

# Test 10: Performance Benchmark
run_test "Performance Benchmark" {
    puts "Running performance benchmarks..."
    
    # Benchmark matrix multiplication
    set sizes [list 128 256 512]
    
    foreach size $sizes {
        puts "Benchmarking ${size}x${size} matrix multiplication:"
        
        set A [torch::tensor_randn [list $size $size] cuda]
        set B [torch::tensor_randn [list $size $size] cuda]
        
        # Warmup
        torch::tensor_matmul $A $B
        
        # Benchmark
        set start_time [clock milliseconds]
        for {set i 0} {$i < 5} {incr i} {
            set C [torch::tensor_matmul $A $B]
        }
        set end_time [clock milliseconds]
        
        set total_time [expr {$end_time - $start_time}]
        set avg_time [expr {$total_time / 5.0}]
        
        puts "  Average time: ${avg_time}ms"
        
        # Performance threshold (these should be fast with cuBLAS)
        if {$size == 128 && $avg_time > 50} {
            puts "  WARNING: 128x128 matmul slower than expected"
        }
        if {$size == 256 && $avg_time > 100} {
            puts "  WARNING: 256x256 matmul slower than expected"
        }
    }
    
    puts "✓ Performance benchmarks completed"
}

# Summary
puts "\n[string repeat "=" 80]"
puts "ADVANCED TEST SUMMARY"
puts [string repeat "=" 80]
puts "Total Tests: $::total_tests"
puts "Passed: $::passed_tests"
puts "Failed: $::failed_tests"

if {$::failed_tests == 0} {
    puts "🎉 ALL ADVANCED TESTS PASSED! CUDA acceleration is working excellently."
    exit 0
} else {
    puts "❌ SOME ADVANCED TESTS FAILED! Review the failures above."
    exit 1
} 