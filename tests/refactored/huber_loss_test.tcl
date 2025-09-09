#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test framework
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax
test huber_loss-1.1 {Basic positional syntax with default parameters} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huber_loss $input $target]
    set result_value [torch::tensor_item $result]
    
    # Should compute Huber loss with delta=1.0, reduction=mean and be non-negative
    expr {$result_value >= 0}
} 1

# Test 2: Positional syntax with explicit reduction
test huber_loss-1.2 {Positional syntax with explicit mean reduction} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result1 [torch::huber_loss $input $target 1]
    set result2 [torch::huber_loss $input $target "mean"]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    # Both should give same result
    expr {abs($value1 - $value2) < 1e-6}
} 1

# Test 3: Positional syntax with none reduction
test huber_loss-1.3 {Positional syntax with none reduction} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huber_loss $input $target "none"]
    
    # Should return per-element losses (shape should be 4)
    set shape [torch::tensor_shape $result]
    expr {[lindex [split $shape " "] 0] == 4}
} 1

# Test 4: Positional syntax with custom delta
test huber_loss-1.4 {Positional syntax with custom delta} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huber_loss $input $target "mean" 0.5]
    set result_value [torch::tensor_item $result]
    
    # Should compute loss with delta=0.5 and be non-negative
    expr {$result_value >= 0}
} 1

# Test 5: Named parameter syntax - basic
test huber_loss-2.1 {Named parameter syntax basic functionality} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huber_loss -input $input -target $target]
    set result_value [torch::tensor_item $result]
    
    # Should compute Huber loss with default parameters
    expr {$result_value >= 0}
} 1

# Test 6: Named parameter syntax - explicit reduction
test huber_loss-2.2 {Named parameter syntax with explicit reduction} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huber_loss -input $input -target $target -reduction "sum"]
    set result_value [torch::tensor_item $result]
    
    # Should compute sum reduction
    expr {$result_value >= 0}
} 1

# Test 7: Named parameter syntax - custom delta
test huber_loss-2.3 {Named parameter syntax with custom delta} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huber_loss -input $input -target $target -delta 2.0]
    set result_value [torch::tensor_item $result]
    
    # Should compute loss with delta=2.0
    expr {$result_value >= 0}
} 1

# Test 8: Named parameter syntax - all parameters
test huber_loss-2.4 {Named parameter syntax with all parameters} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huber_loss -input $input -target $target -reduction "none" -delta 0.8]
    
    # Should return per-element losses with delta=0.8
    set shape [torch::tensor_shape $result]
    expr {[lindex [split $shape " "] 0] == 4}
} 1

# Test 9: CamelCase alias basic functionality
test huber_loss-3.1 {CamelCase alias basic functionality} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huberLoss $input $target]
    set result_value [torch::tensor_item $result]
    
    # CamelCase alias should work
    expr {$result_value >= 0}
} 1

# Test 10: CamelCase alias with named parameters
test huber_loss-3.2 {CamelCase alias with named parameters} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    set result [torch::huberLoss -input $input -target $target -reduction "sum"]
    set result_value [torch::tensor_item $result]
    
    # CamelCase alias with named parameters should work
    expr {$result_value >= 0}
} 1

# Test 11: Syntax consistency - both syntaxes give same result
test huber_loss-4.1 {Syntax consistency - both syntaxes give same result} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    # Positional syntax
    set result1 [torch::huber_loss $input $target "mean" 1.5]
    set value1 [torch::tensor_item $result1]
    
    # Named parameter syntax
    set result2 [torch::huber_loss -input $input -target $target -reduction "mean" -delta 1.5]
    set value2 [torch::tensor_item $result2]
    
    # Both should give same result
    expr {abs($value1 - $value2) < 1e-6}
} 1

# Test 12: Mathematical correctness - small delta threshold
test huber_loss-5.1 {Mathematical correctness - small delta behavior} {
    # When |input - target| <= delta, loss should be 0.5 * (input - target)^2
    set input [torch::tensor_create {2.0} float32]
    set target [torch::tensor_create {1.5} float32]  ; # diff = 0.5 < delta = 1.0
    
    set result [torch::huber_loss $input $target "none" 1.0]
    set result_value [torch::tensor_item $result]
    
    # Should be 0.5 * (2.0 - 1.5)^2 = 0.5 * 0.25 = 0.125
    expr {abs($result_value - 0.125) < 1e-6}
} 1

# Test 13: Mathematical correctness - large delta threshold
test huber_loss-5.2 {Mathematical correctness - large delta behavior} {
    # When |input - target| > delta, loss should be delta * (|input - target| - 0.5 * delta)
    set input [torch::tensor_create {3.0} float32]
    set target [torch::tensor_create {1.0} float32]  ; # diff = 2.0 > delta = 1.0
    
    set result [torch::huber_loss $input $target "none" 1.0]
    set result_value [torch::tensor_item $result]
    
    # Should be 1.0 * (2.0 - 0.5 * 1.0) = 1.0 * 1.5 = 1.5
    expr {abs($result_value - 1.5) < 1e-6}
} 1

# Test 14: Error handling - missing required parameters
test huber_loss-6.1 {Error handling - missing input parameter} {
    set target [torch::tensor_create {1.0 2.0} float32]
    
    set error_caught 0
    if {[catch {torch::huber_loss -target $target} error]} {
        set error_caught 1
    }
    
    set error_caught
} 1

# Test 15: Error handling - invalid tensor name
test huber_loss-6.2 {Error handling - invalid input tensor name} {
    set target [torch::tensor_create {1.0 2.0} float32]
    
    set error_caught 0
    if {[catch {torch::huber_loss "invalid_tensor" $target} error]} {
        set error_caught 1
    }
    
    set error_caught
} 1

# Test 16: Error handling - unknown parameter
test huber_loss-6.3 {Error handling - unknown parameter} {
    set input [torch::tensor_create {1.0 2.0} float32]
    set target [torch::tensor_create {1.5 1.5} float32]
    
    set error_caught 0
    if {[catch {torch::huber_loss -input $input -target $target -unknown "value"} error]} {
        set error_caught 1
    }
    
    set error_caught
} 1

# Test 17: Multi-dimensional tensors
test huber_loss-7.1 {Multi-dimensional tensors} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]
    
    # Reshape to 2x2 tensors
    set input_2d [torch::tensor_reshape $input {2 2}]
    set target_2d [torch::tensor_reshape $target {2 2}]
    
    set result [torch::huber_loss $input_2d $target_2d "mean"]
    set result_value [torch::tensor_item $result]
    
    # Should handle multi-dimensional input
    expr {$result_value >= 0}
} 1

cleanupTests 