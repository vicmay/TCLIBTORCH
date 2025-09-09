#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Helper function to create a test tensor
proc create_test_tensor {} {
    set tensor [torch::randn -shape {2 3 4 4}]
    return $tensor
}

;# Test 1: Basic positional syntax (backward compatibility)
test local_response_norm-1.1 {Basic positional syntax with default parameters} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm $tensor 5 0.0001 0.75 1.0]
    expr {$result ne ""}
} {1}

;# Test 2: Positional syntax with custom size
test local_response_norm-1.2 {Positional syntax with custom size} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm $tensor 3 0.0001 0.75 1.0]
    expr {$result ne ""}
} {1}

;# Test 3: Positional syntax with custom alpha
test local_response_norm-1.3 {Positional syntax with custom alpha} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm $tensor 5 0.001 0.75 1.0]
    expr {$result ne ""}
} {1}

;# Test 4: Named parameter syntax - basic
test local_response_norm-2.1 {Named parameter syntax with -input and -size} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -input $tensor -size 5]
    expr {$result ne ""}
} {1}

;# Test 5: Named parameter syntax - all parameters
test local_response_norm-2.2 {Named parameter syntax with all parameters} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -input $tensor -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
    expr {$result ne ""}
} {1}

;# Test 6: Named parameter syntax with different order
test local_response_norm-2.3 {Named parameter syntax with different order} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -alpha 0.0001 -input $tensor -beta 0.75 -size 5 -k 1.0]
    expr {$result ne ""}
} {1}

;# Test 7: Named parameter syntax with custom values
test local_response_norm-2.4 {Named parameter syntax with custom values} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -input $tensor -size 3 -alpha 0.001 -beta 0.8 -k 2.0]
    expr {$result ne ""}
} {1}

;# Test 8: camelCase alias with positional syntax
test local_response_norm-3.1 {camelCase alias with positional syntax} {
    set tensor [create_test_tensor]
    set result [torch::localResponseNorm $tensor 5 0.0001 0.75 1.0]
    expr {$result ne ""}
} {1}

;# Test 9: camelCase alias with named parameter syntax
test local_response_norm-3.2 {camelCase alias with named parameter syntax} {
    set tensor [create_test_tensor]
    set result [torch::localResponseNorm -input $tensor -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
    expr {$result ne ""}
} {1}

;# Test 10: Error handling - missing tensor (positional)
test local_response_norm-4.1 {Error handling - missing tensor (positional)} {
    catch {torch::local_response_norm nonexistent_tensor 5 0.0001 0.75 1.0} msg
    expr {[string match "*not found*" $msg] || [string match "*Tensor not found*" $msg]}
} {1}

;# Test 11: Error handling - missing parameter value (named)
test local_response_norm-4.2 {Error handling - missing parameter value (named)} {
    catch {torch::local_response_norm -input} msg
    expr {[string match "*Missing value*" $msg]}
} {1}

;# Test 12: Error handling - unknown parameter (named)
test local_response_norm-4.3 {Error handling - unknown parameter (named)} {
    set tensor [create_test_tensor]
    catch {torch::local_response_norm -unknown_param 10 -input $tensor -size 5} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

;# Test 13: Error handling - invalid size value (named)
test local_response_norm-4.4 {Error handling - invalid size value (named)} {
    set tensor [create_test_tensor]
    catch {torch::local_response_norm -input $tensor -size invalid} msg
    expr {[string match "*Invalid*" $msg]}
} {1}

;# Test 14: Error handling - missing required input parameter (named)
test local_response_norm-4.5 {Error handling - missing required input parameter (named)} {
    catch {torch::local_response_norm -size 5 -alpha 0.0001} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

;# Test 15: Different size values
test local_response_norm-5.1 {Different size values} {
    set tensor [create_test_tensor]
    set result1 [torch::local_response_norm -input $tensor -size 3]
    set result2 [torch::local_response_norm -input $tensor -size 5]
    set result3 [torch::local_response_norm -input $tensor -size 7]
    expr {$result1 ne "" && $result2 ne "" && $result3 ne ""}
} {1}

;# Test 16: Different alpha values
test local_response_norm-5.2 {Different alpha values} {
    set tensor [create_test_tensor]
    set result1 [torch::local_response_norm -input $tensor -size 5 -alpha 0.0001]
    set result2 [torch::local_response_norm -input $tensor -size 5 -alpha 0.001]
    set result3 [torch::local_response_norm -input $tensor -size 5 -alpha 0.01]
    expr {$result1 ne "" && $result2 ne "" && $result3 ne ""}
} {1}

;# Test 17: Different beta values
test local_response_norm-5.3 {Different beta values} {
    set tensor [create_test_tensor]
    set result1 [torch::local_response_norm -input $tensor -size 5 -beta 0.5]
    set result2 [torch::local_response_norm -input $tensor -size 5 -beta 0.75]
    set result3 [torch::local_response_norm -input $tensor -size 5 -beta 1.0]
    expr {$result1 ne "" && $result2 ne "" && $result3 ne ""}
} {1}

;# Test 18: Different k values
test local_response_norm-5.4 {Different k values} {
    set tensor [create_test_tensor]
    set result1 [torch::local_response_norm -input $tensor -size 5 -k 1.0]
    set result2 [torch::local_response_norm -input $tensor -size 5 -k 2.0]
    set result3 [torch::local_response_norm -input $tensor -size 5 -k 0.5]
    expr {$result1 ne "" && $result2 ne "" && $result3 ne ""}
} {1}

;# Test 19: Multiple parameter formats create valid results
test local_response_norm-6.1 {Multiple parameter formats create valid results} {
    set tensor [create_test_tensor]
    set result1 [torch::local_response_norm $tensor 5 0.0001 0.75 1.0]
    set result2 [torch::local_response_norm -input $tensor -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
    set result3 [torch::localResponseNorm $tensor 5 0.0001 0.75 1.0]
    set result4 [torch::localResponseNorm -input $tensor -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
    
    expr {$result1 ne "" && $result2 ne "" && $result3 ne "" && $result4 ne ""}
} {1}

;# Test 20: Default parameter values work correctly
test local_response_norm-6.2 {Default parameter values work correctly} {
    set tensor [create_test_tensor]
    set result1 [torch::local_response_norm -input $tensor -size 5]
    set result2 [torch::local_response_norm -input $tensor -size 5 -alpha 0.0001]
    set result3 [torch::local_response_norm -input $tensor -size 5 -alpha 0.0001 -beta 0.75]
    set result4 [torch::local_response_norm -input $tensor -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
    
    expr {$result1 ne "" && $result2 ne "" && $result3 ne "" && $result4 ne ""}
} {1}

;# Test 21: Edge cases - minimum size
test local_response_norm-7.1 {Edge cases - minimum size} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -input $tensor -size 1]
    expr {$result ne ""}
} {1}

;# Test 22: Edge cases - large size
test local_response_norm-7.2 {Edge cases - large size} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -input $tensor -size 11]
    expr {$result ne ""}
} {1}

;# Test 23: Edge cases - very small alpha
test local_response_norm-7.3 {Edge cases - very small alpha} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -input $tensor -size 5 -alpha 1e-8]
    expr {$result ne ""}
} {1}

;# Test 24: Edge cases - large alpha
test local_response_norm-7.4 {Edge cases - large alpha} {
    set tensor [create_test_tensor]
    set result [torch::local_response_norm -input $tensor -size 5 -alpha 1.0]
    expr {$result ne ""}
} {1}

;# Test 25: Different tensor shapes
test local_response_norm-8.1 {Different tensor shapes} {
    set tensor1 [torch::randn -shape {1 3 32 32}]
    set tensor2 [torch::randn -shape {4 64 16 16}]
    set tensor3 [torch::randn -shape {2 128 8 8}]
    
    set result1 [torch::local_response_norm -input $tensor1 -size 5]
    set result2 [torch::local_response_norm -input $tensor2 -size 5]
    set result3 [torch::local_response_norm -input $tensor3 -size 5]
    
    expr {$result1 ne "" && $result2 ne "" && $result3 ne ""}
} {1}

;# Test 26: Parameter validation consistency
test local_response_norm-8.2 {Parameter validation consistency} {
    set tensor [create_test_tensor]
    set valid1 [catch {torch::local_response_norm $tensor 5 0.0001 0.75 1.0} result1]
    set valid2 [catch {torch::local_response_norm -input $tensor -size 5 -alpha 0.0001 -beta 0.75 -k 1.0} result2]
    set valid3 [catch {torch::localResponseNorm $tensor 5 0.0001 0.75 1.0} result3]
    set valid4 [catch {torch::localResponseNorm -input $tensor -size 5 -alpha 0.0001 -beta 0.75 -k 1.0} result4]
    
    expr {$valid1 == 0 && $valid2 == 0 && $valid3 == 0 && $valid4 == 0}
} {1}

;# Test 27: Typical neural network usage
test local_response_norm-8.3 {Typical neural network usage} {
    set conv_output [torch::randn -shape {8 96 55 55}]
    set norm_output [torch::local_response_norm -input $conv_output -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
    expr {$norm_output ne ""}
} {1}

cleanupTests 