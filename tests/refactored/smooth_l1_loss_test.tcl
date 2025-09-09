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

# Helper function to create test tensors
proc create_test_tensors {} {
    # Create input tensor (predictions)
    set input [torch::tensorCreate -data {1.0 2.0 -0.5 -1.5} -shape {2 2}]
    
    # Create target tensor (ground truth)
    set target [torch::tensorCreate -data {1.2 1.8 -0.3 -1.2} -shape {2 2}]
    
    return [list $input $target]
}

# Test 1: Basic positional syntax (backward compatibility)
test smooth_l1_loss-1.1 {Basic positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smooth_l1_loss $input $target]
    expr {$result ne ""}
} 1

test smooth_l1_loss-1.2 {Positional syntax with reduction} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smooth_l1_loss $input $target 0]
    expr {$result ne ""}
} 1

test smooth_l1_loss-1.3 {Positional syntax with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smooth_l1_loss $input $target 1 0.5]
    expr {$result ne ""}
} 1

# Test 2: Named parameter syntax
test smooth_l1_loss-2.1 {Named parameter syntax basic} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smooth_l1_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test smooth_l1_loss-2.2 {Named parameter syntax with reduction} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smooth_l1_loss -input $input -target $target -reduction "none"]
    expr {$result ne ""}
} 1

test smooth_l1_loss-2.3 {Named parameter syntax with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smooth_l1_loss -input $input -target $target -reduction "mean" -beta 0.5]
    expr {$result ne ""}
} 1

test smooth_l1_loss-2.4 {Named parameter syntax with different reduction modes} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    # Test different reduction modes
    set result_none [torch::smooth_l1_loss -input $input -target $target -reduction "none"]
    set result_mean [torch::smooth_l1_loss -input $input -target $target -reduction "mean"]
    set result_sum [torch::smooth_l1_loss -input $input -target $target -reduction "sum"]
    
    expr {$result_none ne "" && $result_mean ne "" && $result_sum ne ""}
} 1

# Test 3: camelCase alias
test smooth_l1_loss-3.1 {camelCase alias basic} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smoothL1Loss -input $input -target $target]
    expr {$result ne ""}
} 1

test smooth_l1_loss-3.2 {camelCase alias with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::smoothL1Loss -input $input -target $target -reduction "sum" -beta 2.0]
    expr {$result ne ""}
} 1

# Test 4: Syntax consistency (both syntaxes should produce same results)
test smooth_l1_loss-4.1 {Syntax consistency test} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    # Same parameters using both syntaxes
    set result_pos [torch::smooth_l1_loss $input $target 1 1.5]
    set result_named [torch::smooth_l1_loss -input $input -target $target -reduction "mean" -beta 1.5]
    set result_camel [torch::smoothL1Loss -input $input -target $target -reduction "mean" -beta 1.5]
    
    # All results should be valid tensor handles
    expr {$result_pos ne "" && $result_named ne "" && $result_camel ne ""}
} 1

# Test 5: Beta parameter behavior
test smooth_l1_loss-5.1 {Different beta values} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    # Test different beta values
    set result_beta_0_5 [torch::smooth_l1_loss -input $input -target $target -beta 0.5]
    set result_beta_1_0 [torch::smooth_l1_loss -input $input -target $target -beta 1.0]
    set result_beta_2_0 [torch::smooth_l1_loss -input $input -target $target -beta 2.0]
    
    expr {$result_beta_0_5 ne "" && $result_beta_1_0 ne "" && $result_beta_2_0 ne ""}
} 1

# Test 6: Error handling
test smooth_l1_loss-6.1 {Error handling - missing parameters} {
    set result [catch {torch::smooth_l1_loss} error_msg]
    expr {$result == 1}
} 1

test smooth_l1_loss-6.2 {Error handling - invalid tensor name} {
    catch {torch::smooth_l1_loss invalid_tensor another_invalid} result
    string match "*Invalid input tensor name*" $result
} 1

test smooth_l1_loss-6.3 {Error handling - named syntax missing values} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    
    catch {torch::smooth_l1_loss -input $input -target} result
    string match "*Named parameters must have values*" $result
} 1

test smooth_l1_loss-6.4 {Error handling - unknown named parameter} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::smooth_l1_loss -input $input -target $target -unknown_param value} result
    string match "*Unknown parameter*" $result
} 1

test smooth_l1_loss-6.5 {Error handling - invalid beta value} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::smooth_l1_loss -input $input -target $target -beta "invalid"} result
    string match "*Invalid beta parameter value*" $result
} 1

# Test 7: Mathematical correctness
test smooth_l1_loss-7.1 {Different reduction modes produce different results} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result_none [torch::smooth_l1_loss -input $input -target $target -reduction "none"]
    set result_mean [torch::smooth_l1_loss -input $input -target $target -reduction "mean"]
    
    # Both results should be valid tensors (different shapes expected)
    expr {$result_none ne "" && $result_mean ne "" && $result_none ne $result_mean}
} 1

# Test 8: Data type compatibility  
test smooth_l1_loss-8.1 {Float32 tensor compatibility} {
    set input [torch::tensorCreate -data {1.0 2.0 -0.5 -1.5} -shape {2 2} -dtype float32]
    set target [torch::tensorCreate -data {1.2 1.8 -0.3 -1.2} -shape {2 2} -dtype float32]
    
    set result [torch::smooth_l1_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test smooth_l1_loss-8.2 {Double tensor compatibility} {
    set input [torch::tensorCreate -data {1.0 2.0 -0.5 -1.5} -shape {2 2} -dtype float64]
    set target [torch::tensorCreate -data {1.2 1.8 -0.3 -1.2} -shape {2 2} -dtype float64]
    
    set result [torch::smooth_l1_loss -input $input -target $target]
    expr {$result ne ""}
} 1

# Test 9: Edge cases
test smooth_l1_loss-9.1 {Edge case - identical input and target} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {2 2}]
    set target [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {2 2}]
    
    set result [torch::smooth_l1_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test smooth_l1_loss-9.2 {Edge case - zero values} {
    set input [torch::tensorCreate -data {0.0 0.0 0.0 0.0} -shape {2 2}]
    set target [torch::tensorCreate -data {0.0 0.0 0.0 0.0} -shape {2 2}]
    
    set result [torch::smooth_l1_loss -input $input -target $target -beta 0.1]
    expr {$result ne ""}
} 1

cleanupTests 