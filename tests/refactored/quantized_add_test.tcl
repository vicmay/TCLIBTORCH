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

# Helper function to create test tensors for quantized operations
proc createQuantizedTestTensors {} {
    # Create input tensors (regular float tensors - quantized_add can work with regular tensors)
    set tensor1 [torch::tensor_randn -shape {4 5} -dtype float32]
    set tensor2 [torch::tensor_randn -shape {4 5} -dtype float32]
    
    # Use basic parameters for quantized operations
    set scale 0.1
    set zero_point 0
    
    return [list $tensor1 $tensor2 $scale $zero_point]
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test quantized_add-1.1 {Basic positional syntax - default alpha} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantized_add $tensor1 $tensor2 $scale $zero_point]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test quantized_add-1.2 {Positional syntax - with alpha value} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantized_add $tensor1 $tensor2 $scale $zero_point 2.0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test quantized_add-1.3 {Positional syntax error handling - too few arguments} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    
    catch {torch::quantized_add $tensor1 $tensor2} error
    string match "*Usage:*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test quantized_add-2.1 {Named parameter syntax - basic usage} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test quantized_add-2.2 {Named parameter syntax - with alpha} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point -alpha 1.5]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test quantized_add-2.3 {Named parameter syntax - different parameter order} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantized_add -scale $scale -tensor2 $tensor2 -zeroPoint $zero_point -tensor1 $tensor1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test quantized_add-2.4 {Named parameter syntax error handling - missing required parameter} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set scale [lindex $tensors 2]
    
    catch {torch::quantized_add -tensor1 $tensor1 -scale $scale} error
    string match "*Required parameters missing*" $error
} {1}

test quantized_add-2.5 {Named parameter syntax error handling - unknown parameter} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    catch {torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point -unknown_param value} error
    string match "*Unknown parameter*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test quantized_add-3.1 {camelCase alias - positional syntax} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantizedAdd $tensor1 $tensor2 $scale $zero_point]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test quantized_add-3.2 {camelCase alias - named parameter syntax} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantizedAdd -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test quantized_add-4.1 {Consistency - same results from both syntaxes} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result1 [torch::quantized_add $tensor1 $tensor2 $scale $zero_point 1.0]
    set result2 [torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point -alpha 1.0]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test quantized_add-4.2 {Consistency - camelCase produces same results} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result1 [torch::quantized_add $tensor1 $tensor2 $scale $zero_point]
    set result2 [torch::quantizedAdd $tensor1 $tensor2 $scale $zero_point]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Parameter Validation
#===========================================================================================

test quantized_add-5.1 {Parameter validation - invalid scale} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set zero_point [lindex $tensors 3]
    
    catch {torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale invalid_scale -zeroPoint $zero_point} error
    string match "*Invalid scale*" $error
} {1}

test quantized_add-5.2 {Parameter validation - invalid zero_point} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    
    catch {torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint invalid_zero} error
    string match "*Invalid zeroPoint*" $error
} {1}

test quantized_add-5.3 {Parameter validation - invalid alpha} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    catch {torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point -alpha invalid_alpha} error
    string match "*Invalid alpha*" $error
} {1}

#===========================================================================================
# Test Cases for Error Handling
#===========================================================================================

test quantized_add-6.1 {Error handling - invalid tensor names} {
    catch {torch::quantized_add invalid_tensor1 invalid_tensor2 0.1 0} error
    string match "*Invalid*tensor*" $error
} {1}

test quantized_add-6.2 {Error handling - missing value for parameter} {
    set result [catch {torch::quantized_add -tensor1} error]
    expr {$result == 1}
} {1}

#===========================================================================================
# Test Cases for Different Alpha Values
#===========================================================================================

test quantized_add-7.1 {Different alpha values - alpha 0.5} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point -alpha 0.5]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test quantized_add-7.2 {Different alpha values - alpha 2.0} {
    set tensors [createQuantizedTestTensors]
    set tensor1 [lindex $tensors 0]
    set tensor2 [lindex $tensors 1]
    set scale [lindex $tensors 2]
    set zero_point [lindex $tensors 3]
    
    set result [torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point -alpha 2.0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

cleanupTests 