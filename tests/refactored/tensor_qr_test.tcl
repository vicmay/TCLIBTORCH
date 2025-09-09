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

# Test cases for positional syntax
test tensor-qr-1.1 {Basic positional syntax} {
    # Create test matrix
    set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test QR decomposition
    set result [torch::tensor_qr $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list $q_data $r_data]
} {{-0.3162277936935425 -0.9486833214759827 -0.9486833214759827 0.3162277638912201} {-3.1622776985168457 -4.427188396453857 0.0 -0.6324553489685059}}

# Test cases for named parameter syntax
test tensor-qr-2.1 {Named parameter syntax} {
    # Create test matrix
    set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test QR decomposition with named parameters
    set result [torch::tensor_qr -tensor $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list $q_data $r_data]
} {{-0.3162277936935425 -0.9486833214759827 -0.9486833214759827 0.3162277638912201} {-3.1622776985168457 -4.427188396453857 0.0 -0.6324553489685059}}

# Test cases for camelCase alias
test tensor-qr-3.1 {CamelCase alias positional syntax} {
    # Create test matrix
    set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test using camelCase alias
    set result [torch::tensorQr $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list $q_data $r_data]
} {{-0.3162277936935425 -0.9486833214759827 -0.9486833214759827 0.3162277638912201} {-3.1622776985168457 -4.427188396453857 0.0 -0.6324553489685059}}

test tensor-qr-3.2 {CamelCase alias named parameter syntax} {
    # Create test matrix
    set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test using camelCase alias with named parameters
    set result [torch::tensorQr -tensor $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list $q_data $r_data]
} {{-0.3162277936935425 -0.9486833214759827 -0.9486833214759827 0.3162277638912201} {-3.1622776985168457 -4.427188396453857 0.0 -0.6324553489685059}}

# Error handling tests
test tensor-qr-4.1 {Error handling - missing tensor} {
    # Test with non-existent tensor
    set result [catch {torch::tensor_qr nonexistent_tensor} error]
    return [list $result $error]
} {1 {Invalid tensor name}}

test tensor-qr-4.2 {Error handling - insufficient arguments} {
    set result [catch {torch::tensor_qr} error]
    return [list $result $error]
} {1 {Usage: torch::tensor_qr tensor | torch::tensor_qr -tensor tensor}}

test tensor-qr-4.3 {Error handling - too many arguments} {
    set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test with too many arguments
    set result [catch {torch::tensor_qr $matrix extra_arg} error]
    
    return [list $result $error]
} {1 {Usage: torch::tensor_qr tensor}}

test tensor-qr-4.4 {Error handling - unknown named parameter} {
    set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test with unknown parameter
    set result [catch {torch::tensor_qr -tensor $matrix -unknown param} error]
    
    return [list $result $error]
} {1 {Unknown parameter: -unknown. Valid parameters are: -tensor}}

test tensor-qr-4.5 {Error handling - missing value for parameter} {
    set matrix [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    
    # Test with missing value
    set result [catch {torch::tensor_qr -tensor} error]
    
    return [list $result $error]
} {1 {Missing value for parameter}}

# Test with different matrix shapes
test tensor-qr-5.1 {3x3 matrix QR decomposition} {
    set matrix [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}} float32 cpu true]
    set result [torch::tensor_qr $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list [llength $q_data] [llength $r_data]]
} {9 9}

test tensor-qr-5.2 {2x3 matrix QR decomposition} {
    set matrix [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result [torch::tensor_qr $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list [llength $q_data] [llength $r_data]]
} {4 6}

# Test with identity matrix
test tensor-qr-6.1 {Identity matrix QR decomposition} {
    set matrix [torch::tensor_create {{1.0 0.0} {0.0 1.0}} float32 cpu true]
    set result [torch::tensor_qr $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list $q_data $r_data]
} {{1.0 0.0 -0.0 1.0} {1.0 0.0 0.0 1.0}}

# Test with zero matrix
test tensor-qr-7.1 {Zero matrix QR decomposition} {
    set matrix [torch::tensor_create {{0.0 0.0} {0.0 0.0}} float32 cpu true]
    set result [torch::tensor_qr $matrix]
    
    # Parse result to get Q and R tensors
    # Result format: {Q tensor_name R tensor_name}
    set result_str [string trim $result "{}"]
    set parts [split $result_str]
    set q_tensor [lindex $parts 1]
    set r_tensor [lindex $parts 3]
    
    # Get tensor data
    set q_data [torch::tensor_to_list $q_tensor]
    set r_data [torch::tensor_to_list $r_tensor]
    
    return [list $q_data $r_data]
} {{1.0 0.0 -0.0 1.0} {0.0 0.0 0.0 0.0}}

cleanupTests 