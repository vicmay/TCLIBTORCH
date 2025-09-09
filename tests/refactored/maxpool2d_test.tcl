#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to create a standard 4x4 test tensor
proc create_test_tensor {} {
    # Create a test input with known pattern for pooling verification
    # Creates a 4x4 tensor with values: [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -shape {1 1 4 4} -dtype float32]
    return $input
}

# Test 1: Basic positional syntax
test maxpool2d-1.1 {Basic positional syntax} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d $input 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 2: Positional syntax with stride
test maxpool2d-1.2 {Positional syntax with stride} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d $input 2 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 3: Positional syntax with stride and padding
test maxpool2d-1.3 {Positional syntax with stride and padding} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d $input 2 2 1]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 4: Named parameter syntax with basic parameters
test maxpool2d-2.1 {Named parameter syntax} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 5: Named parameter syntax with camelCase kernel_size
test maxpool2d-2.2 {Named parameter syntax with camelCase kernelSize} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernelSize 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 6: Named parameter syntax with stride
test maxpool2d-2.3 {Named parameter syntax with stride} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -stride 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 7: Named parameter syntax with padding
test maxpool2d-2.4 {Named parameter syntax with padding} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -padding 1]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 8: Named parameter syntax with tensor alias
test maxpool2d-2.5 {Named parameter syntax with tensor alias} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -tensor $input -kernel_size 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 9: Named parameter syntax with dilation
test maxpool2d-2.6 {Named parameter syntax with dilation} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -dilation 1]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 10: Named parameter syntax with ceil_mode
test maxpool2d-2.7 {Named parameter syntax with ceil_mode} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -ceil_mode 0]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 11: Named parameter syntax with ceilMode alias
test maxpool2d-2.8 {Named parameter syntax with ceilMode alias} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -ceilMode 0]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 12: CamelCase alias command
test maxpool2d-3.1 {CamelCase alias - torch::maxPool2d} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxPool2d $input 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 13: CamelCase alias with named parameters
test maxpool2d-3.2 {CamelCase alias with named parameters} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxPool2d -input $input -kernelSize 2 -stride 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 14: List-based kernel_size (2D)
test maxpool2d-4.1 {List-based kernel_size} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size {2 2}]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 15: List-based stride (2D)
test maxpool2d-4.2 {List-based stride} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -stride {2 2}]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 16: List-based padding (2D)
test maxpool2d-4.3 {List-based padding} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -padding {1 1}]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 17: List-based dilation (2D)
test maxpool2d-4.4 {List-based dilation} {
    set input [torch::ones {1 1 4 4}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -dilation {1 1}]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 18: Error handling - invalid tensor
test maxpool2d-5.1 {Error handling - invalid tensor} {
    catch {torch::maxpool2d invalid_tensor 2} msg
    return [string match "*Invalid input tensor*" $msg]
} {1}

# Test 19: Error handling - missing kernel_size
test maxpool2d-5.2 {Error handling - missing kernel_size} {
    set input [torch::ones {1 1 2 2}]
    catch {torch::maxpool2d -input $input} msg
    ;# tensors auto-managed
    return [string match "*Required parameters*" $msg]
} {1}

# Test 20: Error handling - unknown parameter
test maxpool2d-5.3 {Error handling - unknown parameter} {
    set input [torch::ones {1 1 2 2}]
    catch {torch::maxpool2d -input $input -kernel_size 2 -unknown_param 1} msg
    ;# tensors auto-managed
    return [string match "*Unknown parameter*" $msg]
} {1}

# Test 21: Error handling - invalid kernel_size
test maxpool2d-5.4 {Error handling - invalid kernel_size} {
    set input [torch::ones {1 1 2 2}]
    catch {torch::maxpool2d -input $input -kernel_size 0} msg
    ;# tensors auto-managed
    return [string match "*stride should not be zero*" $msg]
} {1}

# Test 22: Different data types - float64
test maxpool2d-6.1 {Different data types - float64} {
    set input [torch::ones {1 1 4 4} -dtype float64]
    set result [torch::maxpool2d $input 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 23: Different data types - int32
test maxpool2d-6.2 {Different data types - int32} {
    set input [torch::ones {1 1 4 4} -dtype int32]
    set result [torch::maxpool2d $input 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 24: Larger tensor
test maxpool2d-7.1 {Larger tensor} {
    set input [torch::randn {1 1 8 8}]
    set result [torch::maxpool2d $input 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 25: Batch processing
test maxpool2d-7.2 {Batch processing} {
    set input [torch::randn {2 1 4 4}]
    set result [torch::maxpool2d $input 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 26: Multi-channel input
test maxpool2d-7.3 {Multi-channel input} {
    set input [torch::randn {1 3 4 4}]
    set result [torch::maxpool2d $input 2]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 27: Edge case - ceil_mode true
test maxpool2d-8.1 {Edge case - ceil_mode true} {
    set input [torch::ones {1 1 3 3}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -ceil_mode 1]
    ;# tensors auto-managed
    return "ok"
} {ok}

# Test 28: Complete parameter set
test maxpool2d-8.2 {Complete parameter set} {
    set input [torch::ones {1 1 6 6}]
    set result [torch::maxpool2d -input $input -kernel_size 2 -stride 2 -padding 1 -dilation 1 -ceil_mode 0]
    ;# tensors auto-managed
    return "ok"
} {ok}

cleanupTests