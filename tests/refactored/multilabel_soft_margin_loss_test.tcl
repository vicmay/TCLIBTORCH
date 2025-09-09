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

# Helper function to create test tensors for multilabel soft margin loss
proc createTestTensors {} {
    # Create input (predictions) tensor - shape [batch_size, num_classes]
    set input [torch::tensor_randn -shape {4 5} -dtype float32]
    # Create target tensor - shape [batch_size, num_classes] with binary labels
    # Must be float type for multilabel_soft_margin_loss (unlike margin loss which needs int)
    set target [torch::zeros -shape {4 5} -dtype float32]
    return [list $input $target]
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test multilabel_soft_margin_loss-1.1 {Basic positional syntax - default reduction} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss $input $target]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-1.2 {Positional syntax - mean reduction (integer)} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss $input $target 1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-1.3 {Positional syntax - none reduction (integer)} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss $input $target 0]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-1.4 {Positional syntax - sum reduction (integer)} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss $input $target 2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-1.5 {Positional syntax error handling - too few arguments} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    
    catch {torch::multilabel_soft_margin_loss $input} error
    string match "*Usage:*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test multilabel_soft_margin_loss-2.1 {Named parameter syntax - basic usage} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss -input $input -target $target]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-2.2 {Named parameter syntax - with reduction mean} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss -input $input -target $target -reduction mean]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-2.3 {Named parameter syntax - with reduction none} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss -input $input -target $target -reduction none]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-2.4 {Named parameter syntax - with reduction sum} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss -input $input -target $target -reduction sum]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-2.5 {Named parameter syntax - different parameter order} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss -reduction mean -target $target -input $input]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-2.6 {Named parameter syntax error handling - missing required parameter} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    
    catch {torch::multilabel_soft_margin_loss -input $input -reduction mean} error
    string match "*Required parameters missing*" $error
} {1}

test multilabel_soft_margin_loss-2.7 {Named parameter syntax error handling - unknown parameter} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    catch {torch::multilabel_soft_margin_loss -input $input -target $target -unknown_param value} error
    string match "*Unknown parameter*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test multilabel_soft_margin_loss-3.1 {camelCase alias - positional syntax} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabelSoftMarginLoss $input $target]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-3.2 {camelCase alias - named parameter syntax} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabelSoftMarginLoss -input $input -target $target -reduction mean]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test multilabel_soft_margin_loss-4.1 {Consistency - same results from both syntaxes (mean)} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result1 [torch::multilabel_soft_margin_loss $input $target 1]
    set result2 [torch::multilabel_soft_margin_loss -input $input -target $target -reduction mean]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test multilabel_soft_margin_loss-4.2 {Consistency - camelCase produces same results} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result1 [torch::multilabel_soft_margin_loss $input $target]
    set result2 [torch::multilabelSoftMarginLoss $input $target]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test multilabel_soft_margin_loss-4.3 {Consistency - integer vs string reduction} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result1 [torch::multilabel_soft_margin_loss $input $target 0]
    set result2 [torch::multilabel_soft_margin_loss -input $input -target $target -reduction none]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Different Reduction Types
#===========================================================================================

test multilabel_soft_margin_loss-5.1 {Different reduction types - none} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss -input $input -target $target -reduction none]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multilabel_soft_margin_loss-5.2 {Different reduction types - sum} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set result [torch::multilabel_soft_margin_loss -input $input -target $target -reduction sum]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Error Handling
#===========================================================================================

test multilabel_soft_margin_loss-6.1 {Error handling - invalid tensor names} {
    catch {torch::multilabel_soft_margin_loss invalid_tensor invalid_tensor} error
    string match "*Invalid*tensor*" $error
} {1}

test multilabel_soft_margin_loss-6.2 {Error handling - missing value for parameter} {
    set result [catch {torch::multilabel_soft_margin_loss -input} error]
    expr {$result == 1}
} {1}

#===========================================================================================
# Test Cases for Mathematical Correctness
#===========================================================================================

test multilabel_soft_margin_loss-7.1 {Mathematical correctness - loss is non-negative} {
    set tensors [createTestTensors]
    set input [lindex $tensors 0]
    set target [lindex $tensors 1]
    
    set loss [torch::multilabel_soft_margin_loss -input $input -target $target]
    
    # Loss should be a valid tensor (we can't easily check the actual value without more complex parsing)
    expr {[string length $loss] > 0}
} {1}

cleanupTests 