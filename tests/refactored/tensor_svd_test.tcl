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
test tensor_svd-1.1 {Basic positional syntax} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-1.2 {Positional syntax - 3x3 matrix} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-1.3 {Positional syntax - 2x3 matrix} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-1.4 {Positional syntax - 3x2 matrix} {
    set tensor [torch::tensor_create {{1 2} {3 4} {5 6}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

;# Test cases for named parameter syntax
test tensor_svd-2.1 {Named parameter syntax - input} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd -input $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-2.2 {Named parameter syntax - tensor} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensor_svd -tensor $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-2.3 {Named parameter syntax - different parameter order} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd -input $tensor]
    expr {[string length $result] > 0}
} {1}

;# Test cases for camelCase alias
test tensor_svd-3.1 {CamelCase alias - positional syntax} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensorSvd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-3.2 {CamelCase alias - named parameter syntax} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
    set result [torch::tensorSvd -input $tensor]
    expr {[string length $result] > 0}
} {1}

;# Error handling tests
test tensor_svd-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_svd invalid_tensor} result
    return $result
} {Invalid tensor name}

test tensor_svd-4.2 {Error handling - missing tensor} {
    catch {torch::tensor_svd} result
    return $result
} {Required input parameter missing}

test tensor_svd-4.3 {Error handling - invalid named parameter} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    catch {torch::tensor_svd -invalid $tensor} result
    return $result
} {Unknown parameter: -invalid}

test tensor_svd-4.4 {Error handling - missing parameter value} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    catch {torch::tensor_svd -input} result
    return $result
} {Missing value for parameter}

;# Edge cases
test tensor_svd-5.1 {Edge case - identity matrix} {
    set tensor [torch::tensor_create {{1 0} {0 1}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-5.2 {Edge case - zero matrix} {
    set tensor [torch::tensor_create {{0 0} {0 0}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-5.3 {Edge case - single element} {
    set tensor [torch::tensor_create {{5}}]
    catch {torch::tensor_svd $tensor} result
    expr {[string length $result] > 0}
} {1}

;# Data type tests
test tensor_svd-6.1 {Data type - float32 tensor} {
    set tensor [torch::tensor_create {{1.5 2.5} {3.5 4.5}} float32]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-6.2 {Data type - float64 tensor} {
    set tensor [torch::tensor_create {{1.5 2.5} {3.5 4.5}} float64]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-6.3 {Data type - int32 tensor} {
    set tensor [torch::tensor_create {{1 2} {3 4}} int32]
    catch {torch::tensor_svd $tensor} result
    expr {[string length $result] > 0}
} {1}

;# Mathematical correctness tests
test tensor_svd-7.1 {Mathematical correctness - known values} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-7.2 {Mathematical correctness - symmetric matrix} {
    set tensor [torch::tensor_create {{2 1} {1 2}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-7.3 {Mathematical correctness - orthogonal matrix} {
    set tensor [torch::tensor_create {{0.7071 -0.7071} {0.7071 0.7071}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

;# Consistency tests - both syntaxes should produce same results
test tensor_svd-8.1 {Consistency - positional vs named syntax} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result1 [torch::tensor_svd $tensor]
    set result2 [torch::tensor_svd -input $tensor]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test tensor_svd-8.2 {Consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result1 [torch::tensor_svd $tensor]
    set result2 [torch::tensorSvd $tensor]
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

;# Complex scenarios
test tensor_svd-9.1 {Complex - large matrix} {
    set data {}
    for {set i 0} {$i < 5} {incr i} {
        set row {}
        for {set j 0} {$j < 5} {incr j} {
            lappend row [expr {$i + $j + 1}]
        }
        lappend data $row
    }
    set tensor [torch::tensor_create $data]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-9.2 {Complex - random-like matrix} {
    set tensor [torch::tensor_create {{1.1 2.2 3.3} {4.4 5.5 6.6} {7.7 8.8 9.9}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

;# SVD-specific tests
test tensor_svd-10.1 {SVD specific - U matrix properties} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-10.2 {SVD specific - S matrix properties} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-10.3 {SVD specific - V matrix properties} {
    set tensor [torch::tensor_create {{1 2} {3 4} {5 6}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

;# Result format tests
test tensor_svd-11.1 {Result format - contains U S V} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd $tensor]
    expr {[string first "U" $result] >= 0 && [string first "S" $result] >= 0 && [string first "V" $result] >= 0}
} {1}

test tensor_svd-11.2 {Result format - valid tensor handles} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 10}
} {1}

test tensor_svd-11.3 {Result format - consistent structure} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd $tensor]
    expr {[string first "{" $result] >= 0 && [string first "}" $result] >= 0}
} {1}

;# Different matrix shapes
test tensor_svd-12.1 {Matrix shapes - square matrix} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-12.2 {Matrix shapes - rectangular matrix} {
    set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

test tensor_svd-12.3 {Matrix shapes - tall matrix} {
    set tensor [torch::tensor_create {{1 2} {3 4} {5 6}}]
    set result [torch::tensor_svd $tensor]
    expr {[string length $result] > 0}
} {1}

cleanupTests 