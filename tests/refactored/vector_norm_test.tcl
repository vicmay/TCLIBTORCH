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
test vector_norm-1.1 {Basic positional syntax - default ord=2} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0 4.0}]
    set result [torch::vector_norm $tensor1]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 5.477225575051661) < 1e-5}
} 1

test vector_norm-1.2 {Positional syntax with ord parameter} {
    set tensor1 [torch::tensor_create {3.0 4.0}]
    set result [torch::vector_norm $tensor1 1.0]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 7.0) < 1e-5}
} 1

test vector_norm-1.3 {Positional syntax with ord and dim} {
    set tensor1 [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vector_norm $tensor1 2.0 {0}]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 3.1622776601683795) < 1e-5 && abs([lindex $norm_list 1] - 4.47213595499958) < 1e-5}
} 1

test vector_norm-1.4 {Positional syntax with all parameters} {
    set tensor1 [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vector_norm $tensor1 2.0 {0} 1]
    set norm_list [torch::tensor_to_list $result]
    expr {[llength $norm_list] == 2}
} 1

# Test cases for named parameter syntax
test vector_norm-2.1 {Named parameter syntax - basic} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0 4.0}]
    set result [torch::vector_norm -input $tensor1]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 5.477225575051661) < 1e-5}
} 1

test vector_norm-2.2 {Named parameter syntax with ord} {
    set tensor1 [torch::tensor_create {3.0 4.0}]
    set result [torch::vector_norm -input $tensor1 -ord 1.0]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 7.0) < 1e-5}
} 1

test vector_norm-2.3 {Named parameter syntax with dim} {
    set tensor1 [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vector_norm -input $tensor1 -dim {0}]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 3.1622776601683795) < 1e-5 && abs([lindex $norm_list 1] - 4.47213595499958) < 1e-5}
} 1

test vector_norm-2.4 {Named parameter syntax with keepdim} {
    set tensor1 [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vector_norm -input $tensor1 -dim {0} -keepdim 1]
    set shape [torch::tensor_shape $result]
    expr {[lindex $shape 0] == 1 && [lindex $shape 1] == 2}
} 1

test vector_norm-2.5 {Named parameter syntax with all parameters} {
    set tensor1 [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result [torch::vector_norm -input $tensor1 -ord 2.0 -dim {1} -keepdim 0]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 2.23606797749979) < 1e-5 && abs([lindex $norm_list 1] - 5.0) < 1e-5}
} 1

# Test cases for camelCase alias
test vector_norm-3.1 {CamelCase alias - basic functionality} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0 4.0}]
    set result [torch::vectorNorm $tensor1]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 5.477225575051661) < 1e-5}
} 1

test vector_norm-3.2 {CamelCase alias with named parameters} {
    set tensor1 [torch::tensor_create {3.0 4.0}]
    set result [torch::vectorNorm -input $tensor1 -ord 1.0]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 7.0) < 1e-5}
} 1

# Test cases for error handling
test vector_norm-4.1 {Error handling - invalid tensor} {
    catch {torch::vector_norm "invalid_tensor"} error
    string match "*Invalid input tensor*" $error
} 1

test vector_norm-4.2 {Error handling - missing required parameter} {
    catch {torch::vector_norm -ord 2.0} error
    string match "*Required parameter missing*" $error
} 1

test vector_norm-4.3 {Error handling - invalid ord value} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::vector_norm -input $tensor1 -ord "invalid"} error
    string match "*Invalid ord value*" $error
} 1

test vector_norm-4.4 {Error handling - invalid keepdim value} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::vector_norm -input $tensor1 -keepdim "invalid"} error
    string match "*Invalid keepdim value*" $error
} 1

test vector_norm-4.5 {Error handling - unknown parameter} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::vector_norm -input $tensor1 -unknown param} error
    string match "*Unknown parameter*" $error
} 1

# Test cases for mathematical correctness
test vector_norm-5.1 {Mathematical correctness - L2 norm} {
    set tensor1 [torch::tensor_create {3.0 4.0}]
    set result [torch::vector_norm $tensor1 2.0]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 5.0) < 1e-6}
} 1

test vector_norm-5.2 {Mathematical correctness - L1 norm} {
    set tensor1 [torch::tensor_create {3.0 4.0}]
    set result [torch::vector_norm $tensor1 1.0]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 7.0) < 1e-6}
} 1

test vector_norm-5.3 {Mathematical correctness - infinity norm} {
    set tensor1 [torch::tensor_create {3.0 -4.0 2.0}]
    set result [torch::vector_norm $tensor1 inf]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 4.0) < 1e-6}
} 1

# Test cases for data type support
test vector_norm-6.1 {Data type support - float32} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0} float32]
    set result [torch::vector_norm $tensor1]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 3.7416573867739413) < 1e-5}
} 1

test vector_norm-6.2 {Data type support - float64} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0} float64]
    set result [torch::vector_norm $tensor1]
    ;# Just check that the result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} 1

# Test cases for edge cases
test vector_norm-7.1 {Edge case - zero vector} {
    set tensor1 [torch::tensor_create {0.0 0.0 0.0}]
    set result [torch::vector_norm $tensor1]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 0.0) < 1e-10}
} 1

test vector_norm-7.2 {Edge case - single element} {
    set tensor1 [torch::tensor_create {5.0}]
    set result [torch::vector_norm $tensor1]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 5.0) < 1e-10}
} 1

test vector_norm-7.3 {Edge case - large values} {
    set tensor1 [torch::tensor_create {1e6 2e6 3e6}]
    set result [torch::vector_norm $tensor1]
    set norm_list [torch::tensor_to_list $result]
    expr {abs([lindex $norm_list 0] - 3741657.3867739413) < 1e2}
} 1

# Test cases for syntax consistency
test vector_norm-8.1 {Syntax consistency - positional vs named} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0}]
    set result1 [torch::vector_norm $tensor1 2.0]
    set result2 [torch::vector_norm -input $tensor1 -ord 2.0]
    set norm1 [torch::tensor_to_list $result1]
    set norm2 [torch::tensor_to_list $result2]
    expr {abs([lindex $norm1 0] - [lindex $norm2 0]) < 1e-10}
} 1

test vector_norm-8.2 {Syntax consistency - original vs camelCase} {
    set tensor1 [torch::tensor_create {1.0 2.0 3.0}]
    set result1 [torch::vector_norm $tensor1]
    set result2 [torch::vectorNorm $tensor1]
    set norm1 [torch::tensor_to_list $result1]
    set norm2 [torch::tensor_to_list $result2]
    expr {abs([lindex $norm1 0] - [lindex $norm2 0]) < 1e-10}
} 1

test vector_norm-8.3 {Syntax consistency - all combinations} {
    set tensor1 [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
    set result1 [torch::vector_norm $tensor1 2.0 {0} 1]
    set result2 [torch::vector_norm -input $tensor1 -ord 2.0 -dim {0} -keepdim 1]
    set result3 [torch::vectorNorm $tensor1 2.0 {0} 1]
    set result4 [torch::vectorNorm -input $tensor1 -ord 2.0 -dim {0} -keepdim 1]
    
    set norm1 [torch::tensor_to_list $result1]
    set norm2 [torch::tensor_to_list $result2]
    set norm3 [torch::tensor_to_list $result3]
    set norm4 [torch::tensor_to_list $result4]
    
    expr {abs([lindex $norm1 0] - [lindex $norm2 0]) < 1e-10 && 
          abs([lindex $norm1 0] - [lindex $norm3 0]) < 1e-10 && 
          abs([lindex $norm1 0] - [lindex $norm4 0]) < 1e-10}
} 1

cleanupTests 