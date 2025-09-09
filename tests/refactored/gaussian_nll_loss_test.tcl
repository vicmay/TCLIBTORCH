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
test gaussian_nll_loss-1.1 {Basic positional syntax with default parameters} {
    set input [torch::tensor_create {1.0 2.0 0.5 1.5} -dtype float32]
    set target [torch::tensor_create {0.8 1.8 0.3 1.2} -dtype float32]
    set var [torch::tensor_create {0.1 0.2 0.1 0.15} -dtype float32]
    set result [torch::gaussian_nll_loss $input $target $var]
    set loss_val [torch::tensor_item $result]
    expr {$loss_val > -5.0 && $loss_val < 10.0}
} {1}

# Test 2: Named parameter syntax with defaults
test gaussian_nll_loss-2.1 {Named parameter syntax with defaults} {
    set input [torch::tensor_create {1.0 2.0 0.5 1.5} -dtype float32]
    set target [torch::tensor_create {0.8 1.8 0.3 1.2} -dtype float32]
    set var [torch::tensor_create {0.1 0.2 0.1 0.15} -dtype float32]
    set result [torch::gaussian_nll_loss -input $input -target $target -var $var]
    set loss_val [torch::tensor_item $result]
    expr {$loss_val > -5.0 && $loss_val < 10.0}
} {1}

# Test 3: CamelCase alias functionality
test gaussian_nll_loss-3.1 {CamelCase alias functionality} {
    set input [torch::tensor_create {1.0 2.0 0.5 1.5} -dtype float32]
    set target [torch::tensor_create {0.8 1.8 0.3 1.2} -dtype float32]
    set var [torch::tensor_create {0.1 0.2 0.1 0.15} -dtype float32]
    set result [torch::gaussianNllLoss -input $input -target $target -var $var]
    set loss_val [torch::tensor_item $result]
    expr {$loss_val > -5.0 && $loss_val < 10.0}
} {1}

# Test 4: Custom eps parameter
test gaussian_nll_loss-4.1 {Custom eps parameter} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {0.8 1.8} -dtype float32]
    set var [torch::tensor_create {1e-8 1e-7} -dtype float32]
    set result1 [torch::gaussian_nll_loss -input $input -target $target -var $var -eps 1e-6]
    set result2 [torch::gaussian_nll_loss -input $input -target $target -var $var -eps 1e-4]
    set loss1 [torch::tensor_item $result1]
    set loss2 [torch::tensor_item $result2]
    expr {$loss1 != $loss2 && $loss1 > 0.0 && $loss2 > 0.0}
} {1}

# Test 5: Full parameter flag
test gaussian_nll_loss-5.1 {Full parameter flag} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {0.8 1.8} -dtype float32]
    set var [torch::tensor_create {0.1 0.2} -dtype float32]
    set result1 [torch::gaussian_nll_loss -input $input -target $target -var $var -full 0]
    set result2 [torch::gaussian_nll_loss -input $input -target $target -var $var -full 1]
    set loss1 [torch::tensor_item $result1]
    set loss2 [torch::tensor_item $result2]
    expr {$loss2 > $loss1}
} {1}

# Test 6: Reduction option 'none'
test gaussian_nll_loss-6.1 {Reduction option 'none'} {
    set input [torch::tensor_create {1.0 2.0 0.5 1.5} -dtype float32]
    set target [torch::tensor_create {0.8 1.8 0.3 1.2} -dtype float32]
    set var [torch::tensor_create {0.1 0.2 0.1 0.15} -dtype float32]
    set result [torch::gaussian_nll_loss -input $input -target $target -var $var -reduction none]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 1 && [lindex $shape 0] == 4}
} {1}

# Test 7: Reduction option 'sum'
test gaussian_nll_loss-7.1 {Reduction option 'sum'} {
    set input [torch::tensor_create {1.0 2.0 0.5 1.5} -dtype float32]
    set target [torch::tensor_create {0.8 1.8 0.3 1.2} -dtype float32]
    set var [torch::tensor_create {0.1 0.2 0.1 0.15} -dtype float32]
    set result_mean [torch::gaussian_nll_loss -input $input -target $target -var $var -reduction mean]
    set result_sum [torch::gaussian_nll_loss -input $input -target $target -var $var -reduction sum]
    set loss_mean [torch::tensor_item $result_mean]
    set loss_sum [torch::tensor_item $result_sum]
    expr {abs($loss_sum - 4.0 * $loss_mean) < 0.01}
} {1}

# Test 8: Positional syntax with all parameters
test gaussian_nll_loss-8.1 {Positional syntax with all parameters (integer reduction)} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {0.8 1.8} -dtype float32]
    set var [torch::tensor_create {0.1 0.2} -dtype float32]
    # reduction 0 = none
    set result [torch::gaussian_nll_loss $input $target $var 1 1e-5 0]
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] == 1 && [lindex $shape 0] == 2}
} {1}

# Test 9: Error handling - invalid input tensor
test gaussian_nll_loss-9.1 {Error handling - invalid input tensor} {
    set target [torch::tensor_create {0.8 1.8} -dtype float32]
    set var [torch::tensor_create {0.1 0.2} -dtype float32]
    set result [catch {torch::gaussian_nll_loss invalid_tensor $target $var} error]
    expr {$result == 1 && [string match "*Invalid input tensor*" $error]}
} {1}

# Test 10: Error handling - invalid target tensor
test gaussian_nll_loss-10.1 {Error handling - invalid target tensor} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set var [torch::tensor_create {0.1 0.2} -dtype float32]
    set result [catch {torch::gaussian_nll_loss $input invalid_tensor $var} error]
    expr {$result == 1 && [string match "*Invalid target tensor*" $error]}
} {1}

# Test 11: Error handling - invalid var tensor
test gaussian_nll_loss-11.1 {Error handling - invalid var tensor} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {0.8 1.8} -dtype float32]
    set result [catch {torch::gaussian_nll_loss $input $target invalid_var} error]
    expr {$result == 1 && [string match "*Invalid var tensor*" $error]}
} {1}

# Test 12: Error handling - missing required parameters
test gaussian_nll_loss-12.1 {Error handling - missing required parameters} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set result [catch {torch::gaussian_nll_loss -input $input -target target1} error]
    expr {$result == 1 && [string match "*Required parameters*" $error]}
} {1}

# Test 13: Error handling - invalid reduction type
test gaussian_nll_loss-13.1 {Error handling - invalid reduction type} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {0.8 1.8} -dtype float32]
    set var [torch::tensor_create {0.1 0.2} -dtype float32]
    set result [catch {torch::gaussian_nll_loss -input $input -target $target -var $var -reduction invalid} error]
    expr {$result == 1 && [string match "*Invalid reduction type*" $error]}
} {1}

# Test 14: Syntax consistency between positional and named
test gaussian_nll_loss-14.1 {Syntax consistency between positional and named} {
    set input [torch::tensor_create {1.0 2.0 0.5} -dtype float32]
    set target [torch::tensor_create {0.8 1.8 0.3} -dtype float32]
    set var [torch::tensor_create {0.1 0.2 0.15} -dtype float32]
    # mean reduction
    set result1 [torch::gaussian_nll_loss $input $target $var 0 1e-6 1]
    set result2 [torch::gaussian_nll_loss -input $input -target $target -var $var -full 0 -eps 1e-6 -reduction mean]
    set loss1 [torch::tensor_item $result1]
    set loss2 [torch::tensor_item $result2]
    expr {abs($loss1 - $loss2) < 1e-6}
} {1}

# Test 15: Perfect predictions (zero variance handling)
test gaussian_nll_loss-15.1 {Perfect predictions with eps handling} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {1.0 2.0} -dtype float32]
    set var [torch::tensor_create {1e-10 1e-10} -dtype float32]
    set result [torch::gaussian_nll_loss -input $input -target $target -var $var -eps 1e-6]
    set loss_val [torch::tensor_item $result]
    # Should have finite loss due to eps clamping
    expr {$loss_val > -100.0 && $loss_val < 100.0}
} {1}

# Test 16: Different variance values impact
test gaussian_nll_loss-16.1 {Different variance values impact} {
    set input [torch::tensor_create {1.0 1.0} -dtype float32]
    set target [torch::tensor_create {0.5 0.5} -dtype float32]
    set var1 [torch::tensor_create {0.1 0.1} -dtype float32]
    set var2 [torch::tensor_create {1.0 1.0} -dtype float32]
    set result1 [torch::gaussian_nll_loss -input $input -target $target -var $var1]
    set result2 [torch::gaussian_nll_loss -input $input -target $target -var $var2]
    set loss1 [torch::tensor_item $result1]
    set loss2 [torch::tensor_item $result2]
    # Different variance values should give different loss values
    expr {$loss1 != $loss2}
} {1}

cleanupTests 