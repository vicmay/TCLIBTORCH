#!/usr/bin/env tclsh
# Basic Functionality Tests for LibTorch TCL Extension
# These tests are designed to reveal problems clearly without hiding errors

puts [string repeat "=" 80]
puts "LibTorch TCL Extension - Basic Functionality Tests"
puts [string repeat "=" 80]

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

# Test 1: Library Loading
run_test "Library Loading" {
    puts "Attempting to load LibTorch TCL extension..."
    load ../build/libtorchtcl.so
    puts "✓ Library loaded successfully"
}

# Test 2: CUDA Detection
run_test "CUDA Detection" {
    puts "Checking CUDA availability..."
    set cuda_available [torch::cuda_is_available]
    puts "CUDA Available: $cuda_available"
    
    if {$cuda_available != 1} {
        error "CUDA should be available but reports: $cuda_available"
    }
    
    set device_count [torch::cuda_device_count]
    puts "CUDA Device Count: $device_count"
    
    if {$device_count < 1} {
        error "Should have at least 1 CUDA device, found: $device_count"
    }
    puts "✓ CUDA detection working correctly"
}

# Test 3: Basic Tensor Creation (CPU)
run_test "Basic Tensor Creation (CPU)" {
    puts "Creating basic tensor on CPU..."
    set t1 [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu 0]
    puts "Created tensor: $t1"
    
    set printed [torch::tensor_print $t1]
    puts "Tensor contents: $printed"
    
    if {[string length $printed] < 5} {
        error "Tensor print output too short: '$printed'"
    }
    puts "✓ CPU tensor creation successful"
}

# Test 4: Basic Tensor Creation (CUDA)
run_test "Basic Tensor Creation (CUDA)" {
    puts "Creating basic tensor on CUDA..."
    set t1 [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cuda 0]
    puts "Created CUDA tensor: $t1"
    
    set printed [torch::tensor_print $t1]
    puts "CUDA tensor contents: $printed"
    
    # Check if output indicates CUDA
    if {![string match "*CUDA*" $printed]} {
        error "Tensor should be on CUDA but output doesn't indicate CUDA: '$printed'"
    }
    puts "✓ CUDA tensor creation successful"
}

# Test 5: Tensor Device Verification
run_test "Tensor Device Verification" {
    puts "Testing tensor device detection..."
    
    set cpu_tensor [torch::tensor_create {1.0 2.0} float32 cpu 0]
    set cuda_tensor [torch::tensor_create {3.0 4.0} float32 cuda 0]
    
    set cpu_device [torch::tensor_device $cpu_tensor]
    set cuda_device [torch::tensor_device $cuda_tensor]
    
    puts "CPU tensor device: $cpu_device"
    puts "CUDA tensor device: $cuda_device"
    
    if {![string match "*cpu*" $cpu_device]} {
        error "CPU tensor device should contain 'cpu': '$cpu_device'"
    }
    
    if {![string match "*cuda*" $cuda_device]} {
        error "CUDA tensor device should contain 'cuda': '$cuda_device'"
    }
    puts "✓ Device verification successful"
}

# Test 6: Basic Tensor Operations
run_test "Basic Tensor Operations" {
    puts "Testing basic tensor arithmetic..."
    
    set a [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cuda 0]
    set b [torch::tensor_create {5.0 6.0 7.0 8.0} float32 cuda 0]
    
    puts "Tensor A: [torch::tensor_print $a]"
    puts "Tensor B: [torch::tensor_print $b]"
    
    set sum [torch::tensor_add $a $b]
    puts "A + B: [torch::tensor_print $sum]"
    
    set diff [torch::tensor_sub $a $b]
    puts "A - B: [torch::tensor_print $diff]"
    
    set prod [torch::tensor_mul $a $b]
    puts "A * B: [torch::tensor_print $prod]"
    
    # Verify results are on CUDA
    if {![string match "*CUDA*" [torch::tensor_print $sum]]} {
        error "Sum result should be on CUDA"
    }
    puts "✓ Basic tensor operations successful"
}

# Test 7: Tensor Reshaping
run_test "Tensor Reshaping" {
    puts "Testing tensor reshaping..."
    
    set original [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cuda 0]
    puts "Original shape: [torch::tensor_shape $original]"
    
    set reshaped [torch::tensor_reshape $original {2 3}]
    puts "Reshaped to 2x3: [torch::tensor_shape $reshaped]"
    puts "Reshaped tensor: [torch::tensor_print $reshaped]"
    
    set shape [torch::tensor_shape $reshaped]
    if {![string match "*2*3*" $shape]} {
        error "Reshaped tensor should have shape containing 2 and 3: '$shape'"
    }
    puts "✓ Tensor reshaping successful"
}

# Test 8: Matrix Multiplication (cuBLAS test)
run_test "Matrix Multiplication (cuBLAS)" {
    puts "Testing matrix multiplication on CUDA (cuBLAS)..."
    
    set A [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cuda 0]
    set A_matrix [torch::tensor_reshape $A {2 2}]
    
    set B [torch::tensor_create {5.0 6.0 7.0 8.0} float32 cuda 0]
    set B_matrix [torch::tensor_reshape $B {2 2}]
    
    puts "Matrix A: [torch::tensor_print $A_matrix]"
    puts "Matrix B: [torch::tensor_print $B_matrix]"
    
    set result [torch::tensor_matmul $A_matrix $B_matrix]
    puts "A × B: [torch::tensor_print $result]"
    
    # Verify result is on CUDA
    if {![string match "*CUDA*" [torch::tensor_print $result]]} {
        error "Matrix multiplication result should be on CUDA"
    }
    puts "✓ Matrix multiplication (cuBLAS) successful"
}

# Test 9: Neural Network Layer Creation
run_test "Neural Network Layer Creation" {
    puts "Testing neural network layer creation..."
    
    set linear_layer [torch::linear 4 2]
    puts "Created linear layer: $linear_layer"
    
    set conv_layer [torch::conv2d 1 8 3]
    puts "Created conv2d layer: $conv_layer"
    
    if {[string length $linear_layer] < 5} {
        error "Linear layer handle too short: '$linear_layer'"
    }
    
    if {[string length $conv_layer] < 5} {
        error "Conv2d layer handle too short: '$conv_layer'"
    }
    puts "✓ Neural network layer creation successful"
}

# Test 10: Memory Management
run_test "Memory Management" {
    puts "Testing memory management..."
    
    # Create multiple tensors
    for {set i 0} {$i < 10} {incr i} {
        set t [torch::tensor_create [list [expr {$i * 1.0}] [expr {$i * 2.0}]] float32 cuda 0]
        puts "Created tensor $i: $t"
    }
    
    # Check CUDA memory
    set memory_info [torch::cuda_memory_info]
    puts "CUDA memory info: $memory_info"
    
    if {[string length $memory_info] < 5} {
        error "CUDA memory info should provide details: '$memory_info'"
    }
    puts "✓ Memory management test successful"
}

# Test 11: Error Handling
run_test "Error Handling" {
    puts "Testing error handling..."
    
    # Test invalid tensor operations
    set valid_result 1
    
    if {![catch {torch::tensor_create {invalid data} float32 cpu 0} error]} {
        error "Should have failed with invalid data"
    }
    puts "✓ Correctly caught invalid tensor creation: $error"
    
    if {![catch {torch::tensor_add "invalid" "handles"} error]} {
        error "Should have failed with invalid tensor handles"
    }
    puts "✓ Correctly caught invalid tensor operation: $error"
    
    puts "✓ Error handling working correctly"
}

# Test 12: Data Type Verification
run_test "Data Type Verification" {
    puts "Testing data type handling..."
    
    set float_tensor [torch::tensor_create {1.5 2.5 3.5} float32 cuda 0]
    set float_dtype [torch::tensor_dtype $float_tensor]
    puts "Float tensor dtype: $float_dtype"
    
    if {![string match "*Float*" $float_dtype]} {
        error "Float tensor should have Float dtype: '$float_dtype'"
    }
    puts "✓ Data type verification successful"
}

# Summary
puts "\n[string repeat "=" 80]"
puts "TEST SUMMARY"
puts [string repeat "=" 80]
puts "Total Tests: $::total_tests"
puts "Passed: $::passed_tests"
puts "Failed: $::failed_tests"

if {$::failed_tests == 0} {
    puts "🎉 ALL TESTS PASSED! Basic functionality is working correctly."
    exit 0
} else {
    puts "❌ SOME TESTS FAILED! Review the failures above."
    exit 1
} 