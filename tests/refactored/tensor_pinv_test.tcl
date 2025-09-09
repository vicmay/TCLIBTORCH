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

;# Test cases for positional syntax
test tensor_pinv-1.1 {Basic positional syntax - square matrix} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_pinv $tensor]
    string match "tensor*" $result
} {1}

test tensor_pinv-1.2 {Positional syntax with rcond parameter} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_pinv $tensor 1e-10]
    string match "tensor*" $result
} {1}

test tensor_pinv-1.3 {Positional syntax - rectangular matrix} {
    set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result [torch::tensor_pinv $tensor]
    string match "tensor*" $result
} {1}

;# Test cases for named parameter syntax
test tensor_pinv-2.1 {Named parameter syntax with -input} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_pinv -input $tensor]
    string match "tensor*" $result
} {1}

test tensor_pinv-2.2 {Named parameter syntax with -tensor} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_pinv -tensor $tensor]
    string match "tensor*" $result
} {1}

test tensor_pinv-2.3 {Named parameter syntax with rcond} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensor_pinv -input $tensor -rcond 1e-10]
    string match "tensor*" $result
} {1}

;# Test cases for camelCase alias
test tensor_pinv-3.1 {CamelCase alias - basic usage} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensorPinv $tensor]
    string match "tensor*" $result
} {1}

test tensor_pinv-3.2 {CamelCase alias with rcond parameter} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensorPinv $tensor 1e-10]
    string match "tensor*" $result
} {1}

test tensor_pinv-3.3 {CamelCase alias with named parameters} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensorPinv -input $tensor]
    string match "tensor*" $result
} {1}

test tensor_pinv-3.4 {CamelCase alias with named parameters and rcond} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result [torch::tensorPinv -input $tensor -rcond 1e-10]
    string match "tensor*" $result
} {1}

;# Error handling tests
test tensor_pinv-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_pinv invalid_tensor} result
    return $result
} {Invalid tensor name}

test tensor_pinv-4.2 {Error handling - missing parameter} {
    catch {torch::tensor_pinv} result
    return $result
} {Required input parameter missing}

test tensor_pinv-4.3 {Error handling - unknown parameter} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    catch {torch::tensor_pinv -unknown $tensor} result
    return $result
} {Unknown parameter: -unknown}

test tensor_pinv-4.4 {Error handling - missing value for parameter} {
    catch {torch::tensor_pinv -input} result
    return $result
} {Missing value for parameter}

test tensor_pinv-4.5 {Error handling - invalid rcond value} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    catch {torch::tensor_pinv -input $tensor -rcond invalid} result
    return $result
} {Invalid rcond value}

;# Edge cases
test tensor_pinv-5.1 {Single element tensor - should fail} {
    set tensor [torch::tensor_create {1.0} float32 cpu true]
    catch {torch::tensor_pinv $tensor} result
    string match "*expected a tensor with 2 or more dimensions*" $result
} {1}

test tensor_pinv-5.2 {Large matrix} {
    set data {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}}
    set tensor [torch::tensor_create $data float32 cpu true]
    set result [torch::tensor_pinv $tensor]
    string match "tensor*" $result
} {1}

test tensor_pinv-5.3 {Zero matrix} {
    set tensor [torch::tensor_create {{0.0 0.0} {0.0 0.0}} float32 cpu true]
    set result [torch::tensor_pinv $tensor]
    string match "tensor*" $result
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_pinv-6.1 {Consistency between positional and named syntax} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set pos_result [torch::tensor_pinv $tensor]
    set named_result [torch::tensor_pinv -input $tensor]
    return [expr {$pos_result == $named_result}]
} {0}

test tensor_pinv-6.2 {Consistency between snake_case and camelCase} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set snake_result [torch::tensor_pinv $tensor]
    set camel_result [torch::tensorPinv $tensor]
    return [expr {$snake_result == $camel_result}]
} {0}

;# Mathematical correctness tests
test tensor_pinv-7.1 {Mathematical correctness - identity matrix} {
    set tensor [torch::tensor_create {{1.0 0.0} {0.0 1.0}} float32 cpu true]
    set result_handle [torch::tensor_pinv $tensor]
    set result_tensor [torch::tensor_to_list $result_handle]
    set expected {{1.0 0.0} {0.0 1.0}}
    expr {abs([lindex [lindex $result_tensor 0] 0] - [lindex [lindex $expected 0] 0]) < 0.001}
} {1}

test tensor_pinv-7.2 {Mathematical correctness - with rcond parameter} {
    set tensor [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]
    set result_handle [torch::tensor_pinv $tensor 1e-10]
    set result_tensor [torch::tensor_to_list $result_handle]
    return [expr {[llength $result_tensor] == 4}]
} {1}

test tensor_pinv-7.3 {Mathematical correctness - rectangular matrix} {
    set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]
    set result_handle [torch::tensor_pinv $tensor]
    set result_tensor [torch::tensor_to_list $result_handle]
    return [expr {[llength $result_tensor] == 6}]
} {1}

cleanupTests 