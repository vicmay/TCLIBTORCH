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
test margin_ranking_loss-1.1 {Basic positional syntax with default parameters} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::margin_ranking_loss $input1 $input2 $target]
    set result_value [torch::tensor_item $result]
    
    # Should compute margin ranking loss and be non-negative
    expr {$result_value >= 0}
} 1

# Test 2: Positional syntax with explicit margin
test margin_ranking_loss-1.2 {Positional syntax with explicit margin} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result1 [torch::margin_ranking_loss $input1 $input2 $target 0.0]
    set result2 [torch::margin_ranking_loss $input1 $input2 $target 1.0]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    # Both should be non-negative
    expr {$value1 >= 0 && $value2 >= 0}
} 1

# Test 3: Positional syntax with reduction
test margin_ranking_loss-1.3 {Positional syntax with different reductions} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result_mean [torch::margin_ranking_loss $input1 $input2 $target 0.0 "mean"]
    set result_sum [torch::margin_ranking_loss $input1 $input2 $target 0.0 "sum"]
    
    set value_mean [torch::tensor_item $result_mean]
    set value_sum [torch::tensor_item $result_sum]
    
    # Sum should be >= mean for multiple elements
    expr {$value_sum >= $value_mean}
} 1

# Test 4: Positional syntax with none reduction
test margin_ranking_loss-1.4 {Positional syntax with none reduction} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::margin_ranking_loss $input1 $input2 $target 0.0 "none"]
    
    # Should return per-element losses (shape should be 3)
    set shape [torch::tensor_shape $result]
    expr {[lindex [split $shape " "] 0] == 3}
} 1

# Test 5: Named parameter syntax - basic
test margin_ranking_loss-2.1 {Named parameter syntax basic functionality} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target]
    set result_value [torch::tensor_item $result]
    
    # Should compute margin ranking loss with default parameters
    expr {$result_value >= 0}
} 1

# Test 6: Named parameter syntax - explicit margin
test margin_ranking_loss-2.2 {Named parameter syntax with explicit margin} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 1.5]
    set result_value [torch::tensor_item $result]
    
    # Should compute margin ranking loss with custom margin
    expr {$result_value >= 0}
} 1

# Test 7: Named parameter syntax - explicit reduction
test margin_ranking_loss-2.3 {Named parameter syntax with explicit reduction} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -reduction "sum"]
    set result_value [torch::tensor_item $result]
    
    # Should compute sum reduction
    expr {$result_value >= 0}
} 1

# Test 8: Named parameter syntax - all parameters
test margin_ranking_loss-2.4 {Named parameter syntax with all parameters} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.8 -reduction "none"]
    
    # Should return per-element losses with margin=0.8
    set shape [torch::tensor_shape $result]
    expr {[lindex [split $shape " "] 0] == 3}
} 1

# Test 9: CamelCase alias basic functionality
test margin_ranking_loss-3.1 {CamelCase alias basic functionality} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::marginRankingLoss $input1 $input2 $target]
    set result_value [torch::tensor_item $result]
    
    # CamelCase alias should work
    expr {$result_value >= 0}
} 1

# Test 10: CamelCase alias with named parameters
test margin_ranking_loss-3.2 {CamelCase alias with named parameters} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    set result [torch::marginRankingLoss -input1 $input1 -input2 $input2 -target $target -margin 0.5]
    set result_value [torch::tensor_item $result]
    
    # CamelCase alias with named parameters should work
    expr {$result_value >= 0}
} 1

# Test 11: Syntax consistency - both syntaxes give same result
test margin_ranking_loss-4.1 {Syntax consistency - both syntaxes give same result} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    # Positional syntax
    set result1 [torch::margin_ranking_loss $input1 $input2 $target 1.2 "mean"]
    set value1 [torch::tensor_item $result1]
    
    # Named parameter syntax
    set result2 [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 1.2 -reduction "mean"]
    set value2 [torch::tensor_item $result2]
    
    # Both should give same result
    expr {abs($value1 - $value2) < 1e-6}
} 1

# Test 12: Mathematical correctness - target = 1 (input1 should be ranked higher)
test margin_ranking_loss-5.1 {Mathematical correctness - target = 1 behavior} {
    # When target=1, we want input1 > input2, so loss = max(0, -y + margin) where y = input1 - input2
    set input1 [torch::tensor_create {3.0} float32]  ; # Higher value
    set input2 [torch::tensor_create {1.0} float32]  ; # Lower value
    set target [torch::tensor_create {1} float32]    ; # Want input1 > input2
    
    set result [torch::margin_ranking_loss $input1 $input2 $target 0.0 "none"]
    set result_value [torch::tensor_item $result]
    
    # Since input1 > input2 and target=1, loss should be 0
    expr {$result_value < 1e-6}
} 1

# Test 13: Mathematical correctness - target = -1 (input2 should be ranked higher)
test margin_ranking_loss-5.2 {Mathematical correctness - target = -1 behavior} {
    # When target=-1, we want input2 > input1, so loss = max(0, y + margin) where y = input1 - input2
    set input1 [torch::tensor_create -data {1.0} -dtype float32]  ; # Lower value
    set input2 [torch::tensor_create -data {3.0} -dtype float32]  ; # Higher value
    set target [torch::tensor_create -data {-1} -dtype float32]   ; # Want input2 > input1
    
    set result [torch::margin_ranking_loss $input1 $input2 $target 0.0 "none"]
    set result_value [torch::tensor_item $result]
    
    # Since input2 > input1 and target=-1, loss should be 0
    expr {$result_value < 1e-6}
} 1

# Test 14: Mathematical correctness - margin effect
test margin_ranking_loss-5.3 {Mathematical correctness - margin effect} {
    # Test how margin affects the loss
    set input1 [torch::tensor_create {2.0} float32]
    set input2 [torch::tensor_create {1.5} float32]  ; # diff = 0.5
    set target [torch::tensor_create {1} float32]    ; # Want input1 > input2
    
    # With margin 0.0, loss should be 0 since input1 > input2
    set result1 [torch::margin_ranking_loss $input1 $input2 $target 0.0 "none"]
    set value1 [torch::tensor_item $result1]
    
    # With margin 1.0, loss should be max(0, -(0.5) + 1.0) = 0.5
    set result2 [torch::margin_ranking_loss $input1 $input2 $target 1.0 "none"]
    set value2 [torch::tensor_item $result2]
    
    # First should be 0, second should be positive (around 0.5)
    expr {$value1 < 1e-6 && abs($value2 - 0.5) < 1e-6}
} 1

# Test 15: Error handling - missing required parameters
test margin_ranking_loss-6.1 {Error handling - missing input1 parameter} {
    set input2 [torch::tensor_create {1.0 2.0} float32]
    set target [torch::tensor_create {1 1} float32]
    
    set error_caught 0
    if {[catch {torch::margin_ranking_loss -input2 $input2 -target $target} error]} {
        set error_caught 1
    }
    
    set error_caught
} 1

# Test 16: Error handling - invalid tensor name
test margin_ranking_loss-6.2 {Error handling - invalid input1 tensor name} {
    set input2 [torch::tensor_create {1.0 2.0} float32]
    set target [torch::tensor_create {1 1} float32]
    
    set error_caught 0
    if {[catch {torch::margin_ranking_loss "invalid_tensor" $input2 $target} error]} {
        set error_caught 1
    }
    
    set error_caught
} 1

# Test 17: Error handling - unknown parameter
test margin_ranking_loss-6.3 {Error handling - unknown parameter} {
    set input1 [torch::tensor_create {1.0 2.0} float32]
    set input2 [torch::tensor_create {0.5 1.5} float32]
    set target [torch::tensor_create {1 1} float32]
    
    set error_caught 0
    if {[catch {torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -unknown "value"} error]} {
        set error_caught 1
    }
    
    set error_caught
} 1

# Test 18: Multi-dimensional tensors
test margin_ranking_loss-7.1 {Multi-dimensional tensors} {
    set input1 [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5 3.5} float32]
    set target [torch::tensor_create {1 1 1 1} float32]
    
    # Reshape to 2x2 tensors
    set input1_2d [torch::tensor_reshape $input1 {2 2}]
    set input2_2d [torch::tensor_reshape $input2 {2 2}]
    set target_2d [torch::tensor_reshape $target {2 2}]
    
    set result [torch::margin_ranking_loss $input1_2d $input2_2d $target_2d 0.5]
    set result_value [torch::tensor_item $result]
    
    # Should handle multi-dimensional input
    expr {$result_value >= 0}
} 1

# Test 19: Backward compatibility with integer reduction
test margin_ranking_loss-8.1 {Backward compatibility with integer reduction} {
    set input1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set input2 [torch::tensor_create {0.5 1.5 2.5} float32]
    set target [torch::tensor_create {1 1 1} float32]
    
    # Test integer reduction values (legacy)
    set result1 [torch::margin_ranking_loss $input1 $input2 $target 0.0 1]  ; # mean
    set result2 [torch::margin_ranking_loss $input1 $input2 $target 0.0 "mean"]  ; # string
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    # Both should give same result
    expr {abs($value1 - $value2) < 1e-6}
} 1

cleanupTests 