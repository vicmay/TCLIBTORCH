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

;# ============================================================================
;# TORCH::GRADIENT TESTS
;# Test both positional and named parameter syntax
;# ============================================================================

;# Test torch::gradient - Positional Syntax (Backward Compatibility)
test gradient-1.1 {Basic positional syntax} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradient $input]
    torch::tensor_numel $result
} -result 3

test gradient-1.2 {Positional syntax with dim parameter} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0 11.0 16.0} -shape {2 3} -dtype float32]
    set result [torch::gradient $input {} 0]
    torch::tensor_numel $result
} -result 3

test gradient-1.3 {Positional syntax with spacing parameter (ignored)} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradient $input 1.0]
    torch::tensor_numel $result
} -result 3

;# Test torch::gradient - Named Parameter Syntax
test gradient-2.1 {Named parameter syntax with -input} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradient -input $input]
    torch::tensor_numel $result
} -result 3

test gradient-2.2 {Named parameter syntax with -tensor} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradient -tensor $input]
    torch::tensor_numel $result
} -result 3

test gradient-2.3 {Named parameter syntax with -dim} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0 11.0 16.0} -shape {2 3} -dtype float32]
    set result [torch::gradient -input $input -dim 0]
    torch::tensor_numel $result
} -result 3

test gradient-2.4 {Named parameter syntax with -dimension} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0 11.0 16.0} -shape {2 3} -dtype float32]
    set result [torch::gradient -input $input -dimension 1]
    torch::tensor_numel $result
} -result 4

test gradient-2.5 {Named parameter syntax with -spacing (ignored)} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradient -input $input -spacing 1.0]
    torch::tensor_numel $result
} -result 3

test gradient-2.6 {Named parameter syntax with multiple parameters} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0 11.0 16.0} -shape {2 3} -dtype float32]
    set result [torch::gradient -input $input -dim 0 -spacing 1.0]
    torch::tensor_numel $result
} -result 3

;# Test torch::gradientCmd - camelCase alias
test gradient-3.1 {camelCase alias with positional syntax} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradientCmd $input]
    torch::tensor_numel $result
} -result 3

test gradient-3.2 {camelCase alias with named syntax} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradientCmd -input $input]
    torch::tensor_numel $result
} -result 3

test gradient-3.3 {camelCase alias with dim parameter} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0 11.0 16.0} -shape {2 3} -dtype float32]
    set result [torch::gradientCmd -input $input -dim 1]
    torch::tensor_numel $result
} -result 4

;# Test mathematical correctness - shape preservation
test gradient-4.1 {Mathematical correctness - simple 1D case} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradient $input]
    set shape [torch::tensor_shape $result]
    ;# Gradient should have n-1 elements for 1D input
    expr {[lindex $shape 0] == 3}
} -result 1

test gradient-4.2 {Mathematical correctness - 2D tensor along dim 0} -body {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::gradient $input {} 0]
    set shape [torch::tensor_shape $result]
    ;# Gradient along dim 0 should have shape [1, 3]
    expr {[lindex $shape 0] == 1 && [lindex $shape 1] == 3}
} -result 1

test gradient-4.3 {Mathematical correctness - 2D tensor along dim 1} -body {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::gradient $input {} 1]
    set shape [torch::tensor_shape $result]
    ;# Gradient along dim 1 should have shape [2, 2]
    expr {[lindex $shape 0] == 2 && [lindex $shape 1] == 2}
} -result 1

test gradient-4.4 {Mathematical correctness - named syntax shape preservation} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result [torch::gradient -input $input]
    set shape [torch::tensor_shape $result]
    ;# Gradient should have n-1 elements for 1D input
    expr {[lindex $shape 0] == 3}
} -result 1

;# Test syntax consistency - both syntaxes should produce same results
test gradient-5.1 {Syntax consistency - basic gradient} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result1 [torch::gradient $input]
    set result2 [torch::gradient -input $input]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

test gradient-5.2 {Syntax consistency - with dim parameter} -body {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result1 [torch::gradient $input {} 0]
    set result2 [torch::gradient -input $input -dim 0]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

test gradient-5.3 {Syntax consistency - camelCase alias} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    set result1 [torch::gradient $input]
    set result2 [torch::gradientCmd $input]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

;# Test error handling
test gradient-6.1 {Error handling - missing input} -body {
    catch {torch::gradient} result
    expr {[string match "*Required parameter missing*" $result]}
} -result 1

test gradient-6.2 {Error handling - invalid tensor name} -body {
    catch {torch::gradient "invalid_tensor"} result
    expr {[string match "*Invalid tensor name*" $result]}
} -result 1

test gradient-6.3 {Error handling - invalid dim parameter} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    catch {torch::gradient $input {} "invalid_dim"} result
    expr {[string match "*Invalid dim parameter*" $result]}
} -result 1

test gradient-6.4 {Error handling - unknown parameter} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    catch {torch::gradient -input $input -unknown_param value} result
    expr {[string match "*Unknown parameter*" $result]}
} -result 1

test gradient-6.5 {Error handling - missing parameter value} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    catch {torch::gradient -input} result
    expr {[string match "*Named parameters must come in pairs*" $result]}
} -result 1

test gradient-6.6 {Error handling - invalid dim in named syntax} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]
    catch {torch::gradient -input $input -dim "invalid"} result
    expr {[string match "*Invalid dim parameter*" $result]}
} -result 1

;# Test different data types
test gradient-7.1 {Different data types - float64} -body {
    set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float64]
    set result [torch::gradient $input]
    torch::tensor_numel $result
} -result 3

test gradient-7.2 {Different data types - int32} -body {
    set input [torch::tensor_create -data {1 2 4 7} -shape {4} -dtype int32]
    set result [torch::gradient $input]
    torch::tensor_numel $result
} -result 3

test gradient-7.3 {Different data types - int64} -body {
    set input [torch::tensor_create -data {1 2 4 7} -shape {4} -dtype int64]
    set result [torch::gradient $input]
    torch::tensor_numel $result
} -result 3

;# Test edge cases
test gradient-8.1 {Edge case - small tensor} -body {
    set input [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    set result [torch::gradient $input]
    torch::tensor_numel $result
} -result 1

test gradient-8.2 {Edge case - large tensor} -body {
    set data [list]
    for {set i 0} {$i < 100} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set input [torch::tensor_create -data $data -shape {100} -dtype float32]
    set result [torch::gradient $input]
    torch::tensor_numel $result
} -result 99

test gradient-8.3 {Edge case - multidimensional tensor} -body {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
    set result [torch::gradient $input]
    ;# Default should compute along last dimension
    expr {[torch::tensor_numel $result] > 0}
} -result 1

cleanupTests 