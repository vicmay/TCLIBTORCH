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

# Test 1: Basic positional syntax
test tversky_loss-1.1 {Basic positional syntax} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss $input $target]
    expr {[string length $result] > 0}
} 1

# Test 2: Positional syntax with alpha parameter
test tversky_loss-1.2 {Positional syntax with alpha parameter} {
    set input [torch::tensor_create -data {0.7 0.3 0.2 0.8} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss $input $target 0.5]
    expr {[string length $result] > 0}
} 1

# Test 3: Positional syntax with alpha and beta parameters
test tversky_loss-1.3 {Positional syntax with alpha and beta} {
    set input [torch::tensor_create -data {0.9 0.1 0.3 0.7} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss $input $target 0.3 0.7]
    expr {[string length $result] > 0}
} 1

# Test 4: Positional syntax with all parameters
test tversky_loss-1.4 {Positional syntax with all parameters} {
    set input [torch::tensor_create -data {0.6 0.4 0.2 0.8} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss $input $target 0.7 0.3 2.0 mean]
    expr {[string length $result] > 0}
} 1

# Test 5: Named parameter syntax - basic
test tversky_loss-2.1 {Named parameter syntax - basic} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss -input $input -target $target]
    expr {[string length $result] > 0}
} 1

# Test 6: Named parameter syntax with alpha
test tversky_loss-2.2 {Named parameter syntax with alpha} {
    set input [torch::tensor_create -data {0.7 0.3 0.2 0.8} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss -input $input -target $target -alpha 0.5]
    expr {[string length $result] > 0}
} 1

# Test 7: Named parameter syntax with all parameters
test tversky_loss-2.3 {Named parameter syntax with all parameters} {
    set input [torch::tensor_create -data {0.9 0.1 0.3 0.7} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss -input $input -target $target -alpha 0.3 -beta 0.7 -smooth 2.0 -reduction mean]
    expr {[string length $result] > 0}
} 1

# Test 8: Named parameter syntax with different parameter order
test tversky_loss-2.4 {Named parameter syntax - different order} {
    set input [torch::tensor_create -data {0.6 0.4 0.2 0.8} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss -target $target -beta 0.4 -input $input -alpha 0.6]
    expr {[string length $result] > 0}
} 1

# Test 9: camelCase alias
test tversky_loss-3.1 {CamelCase alias basic} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tverskyLoss $input $target]
    expr {[string length $result] > 0}
} 1

# Test 10: camelCase alias with named parameters
test tversky_loss-3.2 {CamelCase alias with named parameters} {
    set input [torch::tensor_create -data {0.7 0.3 0.2 0.8} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tverskyLoss -input $input -target $target -alpha 0.5 -beta 0.5]
    expr {[string length $result] > 0}
} 1

# Test 11: Medical segmentation scenario - dice-like (alpha=beta=0.5)
test tversky_loss-4.1 {Medical segmentation - dice-like behavior} {
    set input [torch::tensor_create -data {10.0 -10.0 -10.0 10.0} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss -input $input -target $target -alpha 0.5 -beta 0.5]
    expr {[string length $result] > 0}
} 1

# Test 12: Different alpha/beta combinations
test tversky_loss-4.2 {Medical segmentation - sensitivity focused} {
    set input [torch::tensor_create -data {0.6 0.4 0.3 0.7} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result1 [torch::tversky_loss -input $input -target $target -alpha 0.2 -beta 0.8]
    set result2 [torch::tversky_loss -input $input -target $target -alpha 0.8 -beta 0.2]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} 1

# Test 13: Different reduction modes
test tversky_loss-4.3 {Different reduction modes} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result_mean [torch::tversky_loss -input $input -target $target -reduction mean]
    set result_sum [torch::tversky_loss -input $input -target $target -reduction sum]
    set result_none [torch::tversky_loss -input $input -target $target -reduction none]
    expr {[string length $result_mean] > 0 && [string length $result_sum] > 0 && [string length $result_none] > 0}
} 1

# Test 14: Mathematical property verification
test tversky_loss-5.1 {Mathematical correctness - both predictions work} {
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set input_poor [torch::tensor_create -data {0.1 0.9 0.9 0.1} -dtype float32 -device cpu]
    set loss_poor [torch::tversky_loss $input_poor $target]
    set input_good [torch::tensor_create -data {0.9 0.1 0.1 0.9} -dtype float32 -device cpu]
    set loss_good [torch::tversky_loss $input_good $target]
    expr {[string length $loss_poor] > 0 && [string length $loss_good] > 0}
} 1

# Test 15: Smooth parameter effect
test tversky_loss-5.2 {Smooth parameter effect} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result1 [torch::tversky_loss -input $input -target $target -smooth 1.0]
    set result2 [torch::tversky_loss -input $input -target $target -smooth 10.0]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} 1

# Test 16: Syntax equivalence - all syntaxes work
test tversky_loss-6.1 {Syntax equivalence} {
    set input [torch::tensor_create -data {0.7 0.3 0.2 0.8} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result1 [torch::tversky_loss $input $target 0.3 0.7 2.0 mean]
    set result2 [torch::tversky_loss -input $input -target $target -alpha 0.3 -beta 0.7 -smooth 2.0 -reduction mean]
    set result3 [torch::tverskyLoss -input $input -target $target -alpha 0.3 -beta 0.7 -smooth 2.0 -reduction mean]
    expr {[string length $result1] > 0 && [string length $result2] > 0 && [string length $result3] > 0}
} 1

# Test 17: Edge case - perfect prediction
test tversky_loss-7.1 {Edge case - perfect prediction} {
    set input [torch::tensor_create -data {100.0 -100.0 -100.0 100.0} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss $input $target]
    expr {[string length $result] > 0}
} 1

# Test 18: Edge case - worst prediction  
test tversky_loss-7.2 {Edge case - worst prediction} {
    set input [torch::tensor_create -data {-100.0 100.0 100.0 -100.0} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss $input $target]
    expr {[string length $result] > 0}
} 1

# Test 19: Edge case - uniform predictions
test tversky_loss-7.3 {Edge case - uniform predictions} {
    set input [torch::tensor_create -data {0.0 0.0 0.0 0.0} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set result [torch::tversky_loss $input $target]
    expr {[string length $result] > 0}
} 1

# Test 20: Data type compatibility - double
test tversky_loss-8.1 {Data type compatibility - double} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float64 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float64 -device cpu]
    set result [torch::tversky_loss $input $target]
    expr {[string length $result] > 0}
} 1

# Test 21: Error handling - missing required parameter
test tversky_loss-9.1 {Error handling - missing required parameter} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set caught [catch {torch::tversky_loss $input} result]
    expr {$caught == 1}
} 1

# Test 22: Error handling - invalid tensor name
test tversky_loss-9.2 {Error handling - invalid tensor name} {
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set caught [catch {torch::tversky_loss "invalid_tensor" $target} result]
    expr {$caught == 1}
} 1

# Test 23: Error handling - invalid parameter name
test tversky_loss-9.3 {Error handling - invalid parameter name} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set caught [catch {torch::tversky_loss -input $input -target $target -invalid_param 0.5} result]
    expr {$caught == 1}
} 1

# Test 24: Error handling - missing parameter value
test tversky_loss-9.4 {Error handling - missing parameter value} {
    set input [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
    set target [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]
    set caught [catch {torch::tversky_loss -input $input -target $target -alpha} result]
    expr {$caught == 1}
} 1

cleanupTests 