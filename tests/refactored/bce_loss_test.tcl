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
test bce_loss-1.1 {Basic positional syntax} {
    set input [torch::tensor_create {0.8 0.2 0.3 0.9} -dtype float32]
    set target [torch::tensor_create {1.0 0.0 0.0 1.0} -dtype float32]
    set loss [torch::bce_loss $input $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.227) < 0.01}  ; # Expected loss ~0.227
} {1}

# Test 2: Named parameter syntax
test bce_loss-2.1 {Named parameter syntax} {
    set input [torch::tensor_create {0.8 0.2 0.3 0.9} -dtype float32]
    set target [torch::tensor_create {1.0 0.0 0.0 1.0} -dtype float32]
    set loss [torch::bce_loss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.227) < 0.01}  ; # Expected loss ~0.227
} {1}

# Test 3: CamelCase alias
test bce_loss-3.1 {CamelCase alias} {
    set input [torch::tensor_create {0.8 0.2 0.3 0.9} -dtype float32]
    set target [torch::tensor_create {1.0 0.0 0.0 1.0} -dtype float32]
    set loss [torch::bceLoss -input $input -target $target]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.227) < 0.01}  ; # Expected loss ~0.227
} {1}

# Test 4: Positional syntax with reduction
test bce_loss-4.1 {Positional syntax with reduction} {
    set input [torch::tensor_create {0.8 0.2 0.3 0.9} -dtype float32]
    set target [torch::tensor_create {1.0 0.0 0.0 1.0} -dtype float32]
    set loss [torch::bce_loss $input $target none sum]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.908) < 0.01}  ; # Expected sum loss ~0.908
} {1}

# Test 5: Named syntax with reduction
test bce_loss-5.1 {Named syntax with reduction} {
    set input [torch::tensor_create {0.8 0.2 0.3 0.9} -dtype float32]
    set target [torch::tensor_create {1.0 0.0 0.0 1.0} -dtype float32]
    set loss [torch::bce_loss -input $input -target $target -reduction sum]
    
    set result [torch::tensor_item $loss]
    expr {abs($result - 0.908) < 0.01}  ; # Expected sum loss ~0.908
} {1}

# Test 6: Named syntax with weight tensor
test bce_loss-6.1 {Named syntax with weight tensor} {
    set input [torch::tensor_create {0.8 0.2 0.3 0.9} -dtype float32]
    set target [torch::tensor_create {1.0 0.0 0.0 1.0} -dtype float32]
    set weight [torch::tensor_create {2.0 1.0 1.0 2.0} -dtype float32]
    set loss [torch::bce_loss -input $input -target $target -weight $weight]
    
    set result [torch::tensor_item $loss]
    expr {$result > 0.0}  ; # Should be positive loss with weights
} {1}

# Test 7: Error handling - missing required parameter
test bce_loss-7.1 {Error handling - missing required parameter} {
    set input [torch::tensor_create {0.8 0.2} -dtype float32]
    catch {torch::bce_loss -input $input} result
    string match "*Required parameters*" $result
} {1}

# Test 8: Error handling - invalid parameter name
test bce_loss-8.1 {Error handling - invalid parameter name} {
    set input [torch::tensor_create {0.8 0.2} -dtype float32]
    set target [torch::tensor_create {1.0 0.0} -dtype float32]
    catch {torch::bce_loss -input $input -target $target -invalid param} result
    string match "*Unknown parameter*" $result
} {1}

# Test 9: Syntax consistency - both syntaxes produce same result
test bce_loss-9.1 {Syntax consistency} {
    set input [torch::tensor_create {0.6 0.4 0.7 0.3} -dtype float32]
    set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]
    
    set loss1 [torch::bce_loss $input $target]
    set loss2 [torch::bce_loss -input $input -target $target]
    
    set result1 [torch::tensor_item $loss1]
    set result2 [torch::tensor_item $loss2]
    
    expr {abs($result1 - $result2) < 1e-6}
} {1}

# Test 10: Edge case - none reduction
test bce_loss-10.1 {Edge case - none reduction} {
    set input [torch::tensor_create {0.7 0.3} -dtype float32]
    set target [torch::tensor_create {1.0 0.0} -dtype float32]
    set loss [torch::bce_loss -input $input -target $target -reduction none]
    
    set shape [torch::tensor_shape $loss]
    list [llength $shape] [lindex $shape 0]
} {1 2}

cleanupTests 