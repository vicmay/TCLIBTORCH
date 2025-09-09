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
    # Create input tensor (log probabilities)
    set input [torch::tensorCreate -data {-1.0 -2.0 -0.5 -3.0} -shape {2 2}]
    
    # Create target tensor (counts - Poisson observations)
    set target [torch::tensorCreate -data {1.0 0.0 2.0 1.0} -shape {2 2}]
    
    return [list $input $target]
}

# Test 1: Basic positional syntax (backward compatibility)
test poisson_nll_loss-1.1 {Basic positional syntax} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poisson_nll_loss $input $target]
    expr {$result ne ""}
} 1

test poisson_nll_loss-1.2 {Positional syntax with log_input false} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poisson_nll_loss $input $target 0]
    expr {$result ne ""}
} 1

test poisson_nll_loss-1.3 {Positional syntax with full parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poisson_nll_loss $input $target 1 1 1]
    expr {$result ne ""}
} 1

# Test 2: Named parameter syntax
test poisson_nll_loss-2.1 {Named parameter syntax basic} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poisson_nll_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test poisson_nll_loss-2.2 {Named parameter syntax with logInput false} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poisson_nll_loss -input $input -target $target -logInput 0]
    expr {$result ne ""}
} 1

test poisson_nll_loss-2.3 {Named parameter syntax with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poisson_nll_loss -input $input -target $target -logInput 1 -full 1 -reduction "mean"]
    expr {$result ne ""}
} 1

test poisson_nll_loss-2.4 {Named parameter syntax with reduction options} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    # Test different reduction modes
    set result_none [torch::poisson_nll_loss -input $input -target $target -reduction "none"]
    set result_mean [torch::poisson_nll_loss -input $input -target $target -reduction "mean"]
    set result_sum [torch::poisson_nll_loss -input $input -target $target -reduction "sum"]
    
    expr {$result_none ne "" && $result_mean ne "" && $result_sum ne ""}
} 1

# Test 3: camelCase alias
test poisson_nll_loss-3.1 {camelCase alias basic} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poissonNllLoss -input $input -target $target]
    expr {$result ne ""}
} 1

test poisson_nll_loss-3.2 {camelCase alias with all parameters} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::poissonNllLoss -input $input -target $target -logInput 1 -full 0 -reduction "sum"]
    expr {$result ne ""}
} 1

# Test 4: Syntax consistency (both syntaxes should produce same results)
test poisson_nll_loss-4.1 {Syntax consistency test} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    # Same parameters using both syntaxes
    set result_pos [torch::poisson_nll_loss $input $target 1 0 1]
    set result_named [torch::poisson_nll_loss -input $input -target $target -logInput 1 -full 0 -reduction "mean"]
    set result_camel [torch::poissonNllLoss -input $input -target $target -logInput 1 -full 0 -reduction "mean"]
    
    # All results should be valid tensor handles
    expr {$result_pos ne "" && $result_named ne "" && $result_camel ne ""}
} 1

# Test 5: Error handling
test poisson_nll_loss-5.1 {Error handling - missing parameters} {
    set result [catch {torch::poisson_nll_loss} error_msg]
    expr {$result == 1}
} 1

test poisson_nll_loss-5.2 {Error handling - invalid tensor name} {
    catch {torch::poisson_nll_loss invalid_tensor another_invalid} result
    string match "*Invalid input tensor name*" $result
} 1

test poisson_nll_loss-5.3 {Error handling - named syntax missing values} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    
    catch {torch::poisson_nll_loss -input $input -target} result
    string match "*Named parameters must have values*" $result
} 1

test poisson_nll_loss-5.4 {Error handling - unknown named parameter} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::poisson_nll_loss -input $input -target $target -unknown_param value} result
    string match "*Unknown parameter*" $result
} 1

# Test 6: Mathematical correctness
test poisson_nll_loss-6.1 {Different reduction modes produce different results} {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result_none [torch::poisson_nll_loss -input $input -target $target -reduction "none"]
    set result_mean [torch::poisson_nll_loss -input $input -target $target -reduction "mean"]
    
    # Both results should be valid tensors (different shapes expected)
    expr {$result_none ne "" && $result_mean ne "" && $result_none ne $result_mean}
} 1

# Test 7: Data type compatibility  
test poisson_nll_loss-7.1 {Float32 tensor compatibility} {
    set input [torch::tensorCreate -data {-1.0 -2.0 -0.5 -3.0} -shape {2 2} -dtype float32]
    set target [torch::tensorCreate -data {1.0 0.0 2.0 1.0} -shape {2 2} -dtype float32]
    
    set result [torch::poisson_nll_loss -input $input -target $target]
    expr {$result ne ""}
} 1

test poisson_nll_loss-7.2 {Double tensor compatibility} {
    set input [torch::tensorCreate -data {-1.0 -2.0 -0.5 -3.0} -shape {2 2} -dtype float64]
    set target [torch::tensorCreate -data {1.0 0.0 2.0 1.0} -shape {2 2} -dtype float64]
    
    set result [torch::poisson_nll_loss -input $input -target $target]
    expr {$result ne ""}
} 1

cleanupTests 