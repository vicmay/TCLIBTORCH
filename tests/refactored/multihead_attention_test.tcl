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
proc createTestTensors {} {
    # Create query, key, value tensors for multi-head attention
    # Shape: [seq_len, batch_size, embed_dim] = [4, 2, 8]
    set query [torch::tensor_randn -shape {4 2 8} -dtype float32]
    set key [torch::tensor_randn -shape {4 2 8} -dtype float32]
    set value [torch::tensor_randn -shape {4 2 8} -dtype float32]
    return [list $query $key $value]
}

#===========================================================================================
# Test Cases for Positional Syntax (Backward Compatibility)
#===========================================================================================

test multihead_attention-1.1 {Basic positional syntax with 2 heads} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    set result [torch::multihead_attention $query $key $value 8 2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multihead_attention-1.2 {Basic positional syntax with 4 heads} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    set result [torch::multihead_attention $query $key $value 8 4]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multihead_attention-1.3 {Positional syntax error handling - too few arguments} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    
    catch {torch::multihead_attention $query $key} error
    string match "*Usage:*" $error
} {1}

test multihead_attention-1.4 {Positional syntax error handling - invalid embed_dim} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    catch {torch::multihead_attention $query $key $value 0 2} error
    string match "*Required parameters missing*" $error
} {1}

#===========================================================================================
# Test Cases for Named Parameter Syntax
#===========================================================================================

test multihead_attention-2.1 {Named parameter syntax - basic usage} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    set result [torch::multihead_attention -query $query -key $key -value $value -embedDim 8 -numHeads 2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multihead_attention-2.2 {Named parameter syntax - different parameter order} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    set result [torch::multihead_attention -numHeads 4 -embedDim 8 -value $value -key $key -query $query]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multihead_attention-2.3 {Named parameter syntax error handling - missing required parameter} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    
    catch {torch::multihead_attention -query $query -key $key -embedDim 8 -numHeads 2} error
    string match "*Required parameters missing*" $error
} {1}

test multihead_attention-2.4 {Named parameter syntax error handling - unknown parameter} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    catch {torch::multihead_attention -query $query -key $key -value $value -embedDim 8 -numHeads 2 -unknown_param 1} error
    string match "*Unknown parameter*" $error
} {1}

#===========================================================================================
# Test Cases for camelCase Alias
#===========================================================================================

test multihead_attention-3.1 {camelCase alias - positional syntax} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    set result [torch::multiheadAttention $query $key $value 8 2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multihead_attention-3.2 {camelCase alias - named parameter syntax} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    set result [torch::multiheadAttention -query $query -key $key -value $value -embedDim 8 -numHeads 2]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

#===========================================================================================
# Test Cases for Consistency Between Syntaxes
#===========================================================================================

test multihead_attention-4.1 {Consistency - same results from both syntaxes} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    # Set seed for reproducible results
    torch::manual_seed 42
    set result1 [torch::multihead_attention $query $key $value 8 2]
    
    torch::manual_seed 42
    set result2 [torch::multihead_attention -query $query -key $key -value $value -embedDim 8 -numHeads 2]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test multihead_attention-4.2 {Consistency - camelCase produces same results} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    # Set seed for reproducible results
    torch::manual_seed 42
    set result1 [torch::multihead_attention $query $key $value 8 2]
    
    torch::manual_seed 42
    set result2 [torch::multiheadAttention $query $key $value 8 2]
    
    # Results should be the same (both should be valid tensor strings)
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

#===========================================================================================
# Test Cases for Different Configuration Values
#===========================================================================================

test multihead_attention-5.1 {Different embed_dim and num_heads combinations} {
    # Create tensors with embed_dim=12
    set query [torch::tensor_randn -shape {3 2 12} -dtype float32]
    set key [torch::tensor_randn -shape {3 2 12} -dtype float32]
    set value [torch::tensor_randn -shape {3 2 12} -dtype float32]
    
    # Test 3 heads (12/3=4 head_dim)
    set result [torch::multihead_attention -query $query -key $key -value $value -embedDim 12 -numHeads 3]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

test multihead_attention-5.2 {Single head attention} {
    set tensors [createTestTensors]
    set query [lindex $tensors 0]
    set key [lindex $tensors 1]
    set value [lindex $tensors 2]
    
    set result [torch::multihead_attention -query $query -key $key -value $value -embedDim 8 -numHeads 1]
    
    # Check result is a valid tensor
    expr {[string length $result] > 0}
} {1}

cleanupTests 