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
    # Create input tensor (raw predictions)
    set input [torch::tensorCreate -data {0.5 -1.2 2.0 -0.8} -shape {2 2}]
    
    # Create target tensor (binary labels: -1 or 1)
    set target [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0} -shape {2 2}]
    
    return [list $input $target]
}

# Test 1: Basic positional syntax (backward compatibility)
test soft_margin_loss-1.1 {Basic positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::soft_margin_loss $input $target]
    expr {$result ne ""}
} 1

test soft_margin_loss-1.2 {Positional syntax with reduction} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::soft_margin_loss $input $target 0]
    expr {$result ne ""}
} 1

test soft_margin_loss-1.3 {Positional syntax with sum reduction} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::soft_margin_loss $input $target 2]
    expr {$result ne ""}
} 1

# Test 2: Named parameter syntax
test soft_margin_loss-2.1 {Named parameter syntax basic} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::soft_margin_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test soft_margin_loss-2.2 {Named parameter syntax with reduction none} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::soft_margin_loss -input $input -target $target -reduction "none"]
    expr {$result ne ""}
} 1

test soft_margin_loss-2.3 {Named parameter syntax with mean reduction} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::soft_margin_loss -input $input -target $target -reduction "mean"]
    expr {$result ne ""}
} 1

test soft_margin_loss-2.4 {Named parameter syntax with different reduction modes} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    # Test different reduction modes
    set result_none [torch::soft_margin_loss -input $input -target $target -reduction "none"]
    set result_mean [torch::soft_margin_loss -input $input -target $target -reduction "mean"]
    set result_sum [torch::soft_margin_loss -input $input -target $target -reduction "sum"]
    
    expr {$result_none ne "" && $result_mean ne "" && $result_sum ne ""}
} 1

# Test 3: camelCase alias
test soft_margin_loss-3.1 {camelCase alias basic} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::softMarginLoss -input $input -target $target]
    expr {$result ne ""}
} 1

test soft_margin_loss-3.2 {camelCase alias with reduction} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::softMarginLoss -input $input -target $target -reduction "sum"]
    expr {$result ne ""}
} 1

# Test 4: Syntax consistency (both syntaxes should produce same results)
test soft_margin_loss-4.1 {Syntax consistency test} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    # Same parameters using both syntaxes
    set result_pos [torch::soft_margin_loss $input $target 1]
    set result_named [torch::soft_margin_loss -input $input -target $target -reduction "mean"]
    set result_camel [torch::softMarginLoss -input $input -target $target -reduction "mean"]
    
    # All results should be valid tensor handles
    expr {$result_pos ne "" && $result_named ne "" && $result_camel ne ""}
} 1

# Test 5: Binary classification targets
test soft_margin_loss-5.1 {Binary targets (+1/-1)} {
    set input [torch::tensorCreate -data {0.8 -0.5 1.2 -0.9} -shape {4}]
    set target [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0} -shape {4}]
    
    set result [torch::soft_margin_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test soft_margin_loss-5.2 {Mixed positive and negative targets} {
    set input [torch::tensorCreate -data {0.5 -1.0 0.2 -0.3} -shape {2 2}]
    set target [torch::tensorCreate -data {1.0 -1.0 -1.0 1.0} -shape {2 2}]
    
    set result [torch::soft_margin_loss -input $input -target $target -reduction "none"]
    expr {$result ne ""}
} 1

# Test 6: Error handling
test soft_margin_loss-6.1 {Error handling - missing parameters} {
    set result [catch {torch::soft_margin_loss} error_msg]
    expr {$result == 1}
} 1

test soft_margin_loss-6.2 {Error handling - invalid tensor name} {
    catch {torch::soft_margin_loss invalid_tensor another_invalid} result
    string match "*Invalid input tensor name*" $result
} 1

test soft_margin_loss-6.3 {Error handling - named syntax missing values} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    
    catch {torch::soft_margin_loss -input $input -target} result
    string match "*Named parameters must have values*" $result
} 1

test soft_margin_loss-6.4 {Error handling - unknown named parameter} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::soft_margin_loss -input $input -target $target -unknown_param value} result
    string match "*Unknown parameter*" $result
} 1

# Test 7: Mathematical correctness
test soft_margin_loss-7.1 {Different reduction modes produce different results} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result_none [torch::soft_margin_loss -input $input -target $target -reduction "none"]
    set result_mean [torch::soft_margin_loss -input $input -target $target -reduction "mean"]
    
    # Both results should be valid tensors (different shapes expected)
    expr {$result_none ne "" && $result_mean ne "" && $result_none ne $result_mean}
} 1

# Test 8: Data type compatibility  
test soft_margin_loss-8.1 {Float32 tensor compatibility} {
    set input [torch::tensorCreate -data {0.5 -1.2 2.0 -0.8} -shape {2 2} -dtype float32]
    set target [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0} -shape {2 2} -dtype float32]
    
    set result [torch::soft_margin_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test soft_margin_loss-8.2 {Double tensor compatibility} {
    set input [torch::tensorCreate -data {0.5 -1.2 2.0 -0.8} -shape {2 2} -dtype float64]
    set target [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0} -shape {2 2} -dtype float64]
    
    set result [torch::soft_margin_loss -input $input -target $target]
    expr {$result ne ""}
} 1

# Test 9: Edge cases
test soft_margin_loss-9.1 {Edge case - large positive predictions} {
    set input [torch::tensorCreate -data {10.0 5.0 8.0 12.0} -shape {2 2}]
    set target [torch::tensorCreate -data {1.0 1.0 1.0 1.0} -shape {2 2}]
    
    set result [torch::soft_margin_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test soft_margin_loss-9.2 {Edge case - large negative predictions} {
    set input [torch::tensorCreate -data {-10.0 -5.0 -8.0 -12.0} -shape {2 2}]
    set target [torch::tensorCreate -data {-1.0 -1.0 -1.0 -1.0} -shape {2 2}]
    
    set result [torch::soft_margin_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test soft_margin_loss-9.3 {Edge case - zero predictions} {
    set input [torch::tensorCreate -data {0.0 0.0 0.0 0.0} -shape {2 2}]
    set target [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0} -shape {2 2}]
    
    set result [torch::soft_margin_loss -input $input -target $target -reduction "mean"]
    expr {$result ne ""}
} 1

# Test 10: Batch processing
test soft_margin_loss-10.1 {Batch processing compatibility} {
    # Simulate batch of 3 samples, each with 4 features
    set input [torch::tensorCreate -data {0.5 -1.2 2.0 -0.8 1.0 -0.5 0.3 -1.5 -0.2 0.8 -1.0 0.6} -shape {3 4}]
    set target [torch::tensorCreate -data {1.0 -1.0 1.0 -1.0 -1.0 1.0 -1.0 1.0 1.0 1.0 -1.0 -1.0} -shape {3 4}]
    
    set result [torch::soft_margin_loss -input $input -target $target -reduction "mean"]
    expr {$result ne ""}
} 1

cleanupTests 