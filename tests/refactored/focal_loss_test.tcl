#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax with default parameters
test focal_loss-1.1 {Basic positional syntax with default parameters} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result [torch::focal_loss $input $target]
    set loss_val [torch::tensor_item $result]
    expr {$loss_val > 0.0 && $loss_val < 2.0}
} {1}

# Test 2: Named parameter syntax with defaults
test focal_loss-2.1 {Named parameter syntax with defaults} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result [torch::focal_loss -input $input -target $target]
    set loss_val [torch::tensor_item $result]
    expr {$loss_val > 0.0 && $loss_val < 2.0}
} {1}

# Test 3: CamelCase alias functionality
test focal_loss-3.1 {CamelCase alias functionality} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result [torch::focalLoss -input $input -target $target]
    set loss_val [torch::tensor_item $result]
    expr {$loss_val > 0.0 && $loss_val < 2.0}
} {1}

# Test 4: Custom alpha parameter
test focal_loss-4.1 {Custom alpha parameter} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result1 [torch::focal_loss -input $input -target $target -alpha 0.5]
    set result2 [torch::focal_loss -input $input -target $target -alpha 2.0]
    set loss1 [torch::tensor_item $result1]
    set loss2 [torch::tensor_item $result2]
    expr {$loss1 != $loss2 && $loss1 > 0.0 && $loss2 > 0.0}
} {1}

# Test 5: Custom gamma parameter
test focal_loss-5.1 {Custom gamma parameter} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result1 [torch::focal_loss -input $input -target $target -gamma 1.0]
    set result2 [torch::focal_loss -input $input -target $target -gamma 3.0]
    set loss1 [torch::tensor_item $result1]
    set loss2 [torch::tensor_item $result2]
    expr {$loss1 != $loss2 && $loss1 > 0.0 && $loss2 > 0.0}
} {1}

# Test 6: Reduction option 'none'
test focal_loss-6.1 {Reduction option 'none'} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result [torch::focal_loss -input $input -target $target -reduction none]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 1 && [lindex $shape 0] == 2}
} {1}

# Test 7: Reduction option 'sum'
test focal_loss-7.1 {Reduction option 'sum'} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result_mean [torch::focal_loss -input $input -target $target -reduction mean]
    set result_sum [torch::focal_loss -input $input -target $target -reduction sum]
    set loss_mean [torch::tensor_item $result_mean]
    set loss_sum [torch::tensor_item $result_sum]
    expr {$loss_sum > $loss_mean && abs($loss_sum - 2.0 * $loss_mean) < 0.01}
} {1}

# Test 8: Positional syntax with all parameters
test focal_loss-8.1 {Positional syntax with all parameters (integer reduction)} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    # reduction 0 = none
    set result [torch::focal_loss $input $target 0.25 1.5 0]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 1 && [lindex $shape 0] == 2}
} {1}

# Test 9: Error handling - invalid input tensor
test focal_loss-9.1 {Error handling - invalid input tensor} {
    set target [torch::tensor_create {0 1} -dtype int64]
    set result [catch {torch::focal_loss invalid_tensor $target} error]
    expr {$result == 1 && [string match "*Invalid input tensor*" $error]}
} {1}

# Test 10: Error handling - invalid target tensor
test focal_loss-10.1 {Error handling - invalid target tensor} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set result [catch {torch::focal_loss $input invalid_tensor} error]
    expr {$result == 1 && [string match "*Invalid target tensor*" $error]}
} {1}

# Test 11: Error handling - missing required parameters
test focal_loss-11.1 {Error handling - missing required parameters} {
    set result [catch {torch::focal_loss -input tensor1} error]
    expr {$result == 1 && [string match "*Required parameters*" $error]}
} {1}

# Test 12: Error handling - invalid reduction type
test focal_loss-12.1 {Error handling - invalid reduction type} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result [catch {torch::focal_loss -input $input -target $target -reduction invalid} error]
    expr {$result == 1 && [string match "*Invalid reduction type*" $error]}
} {1}

# Test 13: Syntax consistency between positional and named
test focal_loss-13.1 {Syntax consistency between positional and named} {
    set input [torch::tensor_create {1.0 2.0 0.5 0.5 1.5 2.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    # mean reduction
    set result1 [torch::focal_loss $input $target 0.75 1.8 1]
    set result2 [torch::focal_loss -input $input -target $target -alpha 0.75 -gamma 1.8 -reduction mean]
    set loss1 [torch::tensor_item $result1]
    set loss2 [torch::tensor_item $result2]
    expr {abs($loss1 - $loss2) < 1e-6}
} {1}

# Test 14: Perfect predictions (low focal loss)
test focal_loss-14.1 {Perfect predictions (low focal loss)} {
    set input [torch::tensor_create {10.0 0.0 0.0 0.0 0.0 10.0} -dtype float32]
    set input [torch::tensor_reshape $input {2 3}]
    set target [torch::tensor_create {0 2} -dtype int64]
    set result [torch::focal_loss -input $input -target $target -alpha 1.0 -gamma 2.0]
    set loss_val [torch::tensor_item $result]
    # Should be very low for confident predictions
    expr {$loss_val < 0.1}
} {1}

cleanupTests 