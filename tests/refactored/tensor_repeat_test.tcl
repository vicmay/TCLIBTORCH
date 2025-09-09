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
test tensor_repeat-1.1 {Basic positional syntax - 1D tensor} {
    set tensor [torch::tensor_create {1 2 3}]
    set result [torch::tensor_repeat $tensor {2}]
    set shape [torch::tensor_shape $result]
    return $shape
} {6}

test tensor_repeat-1.2 {Basic positional syntax - 2D tensor} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_repeat $tensor {2 3}]
    set shape [torch::tensor_shape $result]
    return $shape
} {4 6}

test tensor_repeat-1.3 {Basic positional syntax - scalar tensor} {
    set tensor [torch::tensor_create 5]
    set result [torch::tensor_repeat $tensor {3 2}]
    set shape [torch::tensor_shape $result]


    return $shape
} {3 2}

;# Test cases for named parameter syntax
test tensor_repeat-2.1 {Named parameter syntax - input and repeats} {
    set tensor [torch::tensor_create {1 2 3}]
    set result [torch::tensor_repeat -input $tensor -repeats {3}]
    set shape [torch::tensor_shape $result]


    return $shape
} {9}

test tensor_repeat-2.2 {Named parameter syntax - tensor and repeats} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensor_repeat -tensor $tensor -repeats {2 2}]
    set shape [torch::tensor_shape $result]


    return $shape
} {4 4}

test tensor_repeat-2.3 {Named parameter syntax - complex repeats} {
    set tensor [torch::tensor_create {1 2}]
    set result [torch::tensor_repeat -input $tensor -repeats {3 2}]
    set shape [torch::tensor_shape $result]


    return $shape
} {3 4}

;# Test cases for camelCase alias
test tensor_repeat-3.1 {CamelCase alias - positional syntax} {
    set tensor [torch::tensor_create {1 2 3}]
    set result [torch::tensorRepeat $tensor {2}]
    set shape [torch::tensor_shape $result]


    return $shape
} {6}

test tensor_repeat-3.2 {CamelCase alias - named parameter syntax} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result [torch::tensorRepeat -input $tensor -repeats {2 3}]
    set shape [torch::tensor_shape $result]


    return $shape
} {4 6}

;# Error handling tests
test tensor_repeat-4.1 {Error handling - invalid tensor name} {
    catch {torch::tensor_repeat invalid_tensor {2 3}} result
    return $result
} {Invalid tensor name}

test tensor_repeat-4.2 {Error handling - missing repeats} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_repeat $tensor} result
    return $result
} {Invalid number of arguments}

test tensor_repeat-4.3 {Error handling - empty repeats} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_repeat $tensor {}} result

    return $result
} {Required parameters missing: input tensor and repeats are required}

test tensor_repeat-4.4 {Error handling - invalid named parameter} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_repeat -invalid $tensor -repeats {2}} result

    return $result
} {Unknown parameter: -invalid}

test tensor_repeat-4.5 {Error handling - missing parameter value} {
    set tensor [torch::tensor_create {1 2 3}]
    catch {torch::tensor_repeat -input $tensor -repeats} result

    return $result
} {Missing value for parameter}

;# Edge cases
test tensor_repeat-5.1 {Edge case - repeat by 1 (no change)} {
    set tensor [torch::tensor_create {1 2 3}]
    set result [torch::tensor_repeat $tensor {1}]
    set shape [torch::tensor_shape $result]


    return $shape
} {3}

test tensor_repeat-5.2 {Edge case - large repeats} {
    set tensor [torch::tensor_create {1 2}]
    set result [torch::tensor_repeat $tensor {10 5}]
    set shape [torch::tensor_shape $result]


    return $shape
} {10 10}

test tensor_repeat-5.3 {Edge case - zero tensor} {
    set tensor [torch::tensor_create 0]
    set result [torch::tensor_repeat $tensor {3 2}]
    set shape [torch::tensor_shape $result]


    return $shape
} {3 2}

;# Data type tests
test tensor_repeat-6.1 {Data type - float tensor} {
    set tensor [torch::tensor_create {1.5 2.5 3.5} float32]
    set result [torch::tensor_repeat $tensor {2}]
    set shape [torch::tensor_shape $result]


    return $shape
} {6}

test tensor_repeat-6.2 {Data type - int tensor} {
    set tensor [torch::tensor_create {1 2 3} int64]
    set result [torch::tensor_repeat $tensor {3}]
    set shape [torch::tensor_shape $result]


    return $shape
} {9}

;# Consistency tests - both syntaxes should produce same results
test tensor_repeat-7.1 {Consistency - positional vs named syntax} {
    set tensor [torch::tensor_create {1 2 3}]
    set result1 [torch::tensor_repeat $tensor {2}]
    set result2 [torch::tensor_repeat -input $tensor -repeats {2}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 == $shape2}]
} {1}

test tensor_repeat-7.2 {Consistency - snake_case vs camelCase} {
    set tensor [torch::tensor_create {{1 2} {3 4}}]
    set result1 [torch::tensor_repeat $tensor {2 2}]
    set result2 [torch::tensorRepeat $tensor {2 2}]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    return [expr {$shape1 == $shape2}]
} {1}

cleanupTests 