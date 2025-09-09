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
test mse_loss-1.1 {Basic positional syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} -dtype float32]
    set target [torch::tensor_create {1.1 1.9 3.2 3.8} -dtype float32]
    set loss [torch::mse_loss $input $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.025) < 0.01}  ; # Expected MSE loss ~0.025
} {1}

# Test 2: Named parameter syntax
test mse_loss-2.1 {Named parameter syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} -dtype float32]
    set target [torch::tensor_create {1.1 1.9 3.2 3.8} -dtype float32]
    set loss [torch::mse_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.025) < 0.01}  ; # Expected MSE loss ~0.025
} {1}

# Test 3: CamelCase alias
test mse_loss-3.1 {CamelCase alias} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} -dtype float32]
    set target [torch::tensor_create {1.1 1.9 3.2 3.8} -dtype float32]
    set loss [torch::mseLoss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.025) < 0.01}  ; # Expected MSE loss ~0.025
} {1}

# Test 4: Positional syntax with mean reduction (default)
test mse_loss-4.1 {Positional syntax with mean reduction} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} -dtype float32]
    set target [torch::tensor_create {2.0 3.0 4.0 5.0} -dtype float32]
    set loss [torch::mse_loss $input $target mean]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 1.0) < 0.01}  ; # Expected mean loss = 1.0
} {1}

# Test 5: Named syntax with sum reduction
test mse_loss-5.1 {Named syntax with sum reduction} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} -dtype float32]
    set target [torch::tensor_create {2.0 3.0 4.0 5.0} -dtype float32]
    set loss [torch::mse_loss -input $input -target $target -reduction sum]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 4.0) < 0.01}  ; # Expected sum loss = 4.0
} {1}

# Test 6: Named syntax with none reduction
test mse_loss-6.1 {Named syntax with none reduction} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {2.0 3.0} -dtype float32]
    set loss [torch::mse_loss -input $input -target $target -reduction none]
    
    # Should return unreduced losses for each element
    set shape [torch::tensor_shape $loss]
    list [llength $shape] [lindex $shape 0]
} {1 2}

# Test 7: Error handling - missing required parameter
test mse_loss-7.1 {Error handling - missing required parameter} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    catch {torch::mse_loss -input $input} result
    string match "*Required parameters*" $result
} {1}

# Test 8: Error handling - invalid parameter name
test mse_loss-8.1 {Error handling - invalid parameter name} {
    set input [torch::tensor_create {1.0 2.0} -dtype float32]
    set target [torch::tensor_create {1.1 1.9} -dtype float32]
    catch {torch::mse_loss -input $input -target $target -invalid param} result
    string match "*Unknown parameter*" $result
} {1}

# Test 9: Syntax consistency - both syntaxes produce same result
test mse_loss-9.1 {Syntax consistency} {
    set input [torch::tensor_create {0.5 1.5 2.5 3.5} -dtype float32]
    set target [torch::tensor_create {0.6 1.4 2.6 3.4} -dtype float32]
    
    set loss1 [torch::mse_loss $input $target]
    set loss2 [torch::mse_loss -input $input -target $target]
    
    set result1 [torch::tensor_item $loss1]
    set result2 [torch::tensor_item $loss2]
    
    expr {abs($result1 - $result2) < 1e-6}
} {1}

# Test 10: Different tensor shapes
test mse_loss-10.1 {Different tensor shapes} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} {2 2} float32]
    set target [torch::tensor_create {1.1 1.9 3.1 3.9} {2 2} float32]
    set loss [torch::mse_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Should be positive loss
} {1}

# Test 11: Zero loss case
test mse_loss-11.1 {Zero loss case} {
    set input [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set target [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
    set loss [torch::mse_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result) < 1e-6}  ; # Should be zero loss
} {1}

# Test 12: Large difference case
test mse_loss-12.1 {Large difference case} {
    set input [torch::tensor_create {0.0 0.0} -dtype float32]
    set target [torch::tensor_create {10.0 10.0} -dtype float32]
    set loss [torch::mse_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 100.0) < 0.01}  ; # Should be 100.0
} {1}

# Test 13: Backward compatibility - old positional syntax still works
test mse_loss-13.1 {Backward compatibility} {
    set input [torch::tensor_create {2.0 4.0} -dtype float32]
    set target [torch::tensor_create {1.0 3.0} -dtype float32]
    
    # This should work exactly as before
    set loss [torch::mse_loss $input $target]
    set result [torch::tensor_item $loss]
    expr {abs($result - 1.0) < 0.01}  ; # (1^2 + 1^2) / 2 = 1.0
} {1}

# Test 14: Large tensor case
test mse_loss-14.1 {Large tensor case} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} -dtype float32]
    set target [torch::tensor_create {1.1 2.1 3.1 4.1 5.1} -dtype float32]
    set loss [torch::mseLoss -input $input -target $target -reduction sum]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.05) < 0.01}  ; # 5 * (0.1^2) = 0.05
} {1}

cleanupTests 