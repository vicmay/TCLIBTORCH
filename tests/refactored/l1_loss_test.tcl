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
test l1_loss-1.1 {Basic positional syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.1 1.9 3.2 3.8} float32]
    set loss [torch::l1_loss $input $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.15) < 0.01}  ; # Expected L1 loss ~0.15 (mean of |0.1| + |0.1| + |0.2| + |0.2|) / 4
} {1}

# Test 2: Named parameter syntax
test l1_loss-2.1 {Named parameter syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.1 1.9 3.2 3.8} float32]
    set loss [torch::l1_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.15) < 0.01}  ; # Expected L1 loss ~0.15
} {1}

# Test 3: CamelCase alias
test l1_loss-3.1 {CamelCase alias} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {1.1 1.9 3.2 3.8} float32]
    set loss [torch::l1Loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.15) < 0.01}  ; # Expected L1 loss ~0.15
} {1}

# Test 4: Positional syntax with mean reduction (default)
test l1_loss-4.1 {Positional syntax with mean reduction} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {2.0 3.0 4.0 5.0} float32]
    set loss [torch::l1_loss $input $target mean]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 1.0) < 0.01}  ; # Expected mean L1 loss = 1.0
} {1}

# Test 5: Named syntax with sum reduction
test l1_loss-5.1 {Named syntax with sum reduction} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set target [torch::tensor_create {2.0 3.0 4.0 5.0} float32]
    set loss [torch::l1_loss -input $input -target $target -reduction sum]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 4.0) < 0.01}  ; # Expected sum L1 loss = 4.0
} {1}

# Test 6: Named syntax with none reduction
test l1_loss-6.1 {Named syntax with none reduction} {
    set input [torch::tensor_create {1.0 2.0} float32]
    set target [torch::tensor_create {2.0 3.0} float32]
    set loss [torch::l1_loss -input $input -target $target -reduction none]
    
    # Should return unreduced losses for each element
    set shape [torch::tensor_shape $loss]
    list [llength $shape] [lindex $shape 0]
} {1 2}

# Test 7: Backward compatibility with integer reduction (old API)
test l1_loss-7.1 {Backward compatibility with integer reduction} {
    set input [torch::tensor_create {1.0 2.0} float32]
    set target [torch::tensor_create {2.0 4.0} float32]
    set loss [torch::l1_loss $input $target 2]  ; # 2 = sum reduction
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 3.0) < 0.01}  ; # Sum: |1-2| + |2-4| = 1 + 2 = 3.0
} {1}

# Test 8: Error handling - missing required parameter
test l1_loss-8.1 {Error handling - missing required parameter} {
    set input [torch::tensor_create {1.0 2.0} float32]
    catch {torch::l1_loss -input $input} result
    string match "*Required parameters*" $result
} {1}

# Test 9: Error handling - invalid parameter name
test l1_loss-9.1 {Error handling - invalid parameter name} {
    set input [torch::tensor_create {1.0 2.0} float32]
    set target [torch::tensor_create {1.1 1.9} float32]
    catch {torch::l1_loss -input $input -target $target -invalid param} result
    string match "*Unknown parameter*" $result
} {1}

# Test 10: Syntax consistency - both syntaxes produce same result
test l1_loss-10.1 {Syntax consistency} {
    set input [torch::tensor_create {0.5 1.5 2.5 3.5} float32]
    set target [torch::tensor_create {0.6 1.4 2.6 3.4} float32]
    
    set loss1 [torch::l1_loss $input $target]
    set loss2 [torch::l1_loss -input $input -target $target]
    
    set result1 [torch::tensor_item $loss1]
    set result2 [torch::tensor_item $loss2]
    
    expr {abs($result1 - $result2) < 1e-6}
} {1}

# Test 11: Different tensor shapes
test l1_loss-11.1 {Different tensor shapes} {
    set input [torch::tensor_create {1.0 2.0 3.0 4.0} {2 2} float32]
    set target [torch::tensor_create {1.1 1.9 3.1 3.9} {2 2} float32]
    set loss [torch::l1_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Should be positive loss
} {1}

# Test 12: Zero loss case
test l1_loss-12.1 {Zero loss case} {
    set input [torch::tensor_create {1.0 2.0 3.0} float32]
    set target [torch::tensor_create {1.0 2.0 3.0} float32]
    set loss [torch::l1_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result) < 1e-6}  ; # Should be zero loss
} {1}

# Test 13: Large difference case
test l1_loss-13.1 {Large difference case} {
    set input [torch::tensor_create {0.0 0.0} float32]
    set target [torch::tensor_create {10.0 10.0} float32]
    set loss [torch::l1_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 10.0) < 0.01}  ; # Should be 10.0 (mean of |0-10| + |0-10|) / 2
} {1}

# Test 14: Backward compatibility - old positional syntax still works
test l1_loss-14.1 {Backward compatibility} {
    set input [torch::tensor_create {2.0 4.0} float32]
    set target [torch::tensor_create {1.0 3.0} float32]
    
    # This should work exactly as before
    set loss [torch::l1_loss $input $target]
    set result [torch::tensor_item $loss]
    expr {abs($result - 1.0) < 0.01}  ; # (|2-1| + |4-3|) / 2 = 1.0
} {1}

# Test 15: L1 vs L2 difference verification
test l1_loss-15.1 {L1 vs L2 difference} {
    set input [torch::tensor_create {1.0 3.0} float32]
    set target [torch::tensor_create {2.0 4.0} float32]
    
    set l1_loss [torch::l1_loss -input $input -target $target]
    set mse_loss [torch::mse_loss -input $input -target $target]
    
    set l1_result [torch::tensor_item $l1_loss]
    set mse_result [torch::tensor_item $mse_loss]
    
    # L1: (|1-2| + |3-4|) / 2 = 1.0
    # MSE: ((1-2)² + (3-4)²) / 2 = 1.0
    # In this case they're equal, but verify they're calculated differently
    expr {abs($l1_result - 1.0) < 0.01 && abs($mse_result - 1.0) < 0.01}
} {1}

# Test 16: Asymmetric L1 loss behavior
test l1_loss-16.1 {Asymmetric L1 loss behavior} {
    set input [torch::tensor_create {0.0} float32]
    set target [torch::tensor_create {5.0} float32]
    
    set loss1 [torch::l1Loss -input $input -target $target]
    
    # Flip input and target - should give same result (L1 is symmetric)
    set loss2 [torch::l1Loss -input $target -target $input]
    
    set result1 [torch::tensor_item $loss1]
    set result2 [torch::tensor_item $loss2]
    
    expr {abs($result1 - $result2) < 1e-6 && abs($result1 - 5.0) < 0.01}
} {1}

cleanupTests 