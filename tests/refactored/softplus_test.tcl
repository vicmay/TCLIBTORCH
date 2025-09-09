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

;# Test 1: Basic positional syntax
test softplus-1.1 {Basic positional syntax} {
    set input [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0} -shape {5} -dtype float32]
    
    set result [torch::softplus $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {5}

;# Test 2: Named parameter syntax
test softplus-2.1 {Named parameter syntax with -input} {
    set input [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0} -shape {5} -dtype float32]
    
    set result [torch::softplus -input $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {5}

;# Test 3: Named parameter syntax with -tensor
test softplus-2.2 {Named parameter syntax with -tensor} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0} -shape {3} -dtype float32]
    
    set result [torch::softplus -tensor $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {3}

;# Test 4: camelCase alias - positional syntax
test softplus-3.1 {camelCase alias - positional syntax} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0 2.0} -shape {4} -dtype float32]
    
    set result [torch::softPlus $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {4}

;# Test 5: camelCase alias - named syntax
test softplus-3.2 {camelCase alias - named syntax} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0 2.0} -shape {4} -dtype float32]
    
    set result [torch::softPlus -input $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {4}

;# Test 6: Mathematical correctness - always positive
test softplus-4.1 {Mathematical correctness - always positive output} {
    set input [torch::tensorCreate -data {-100.0 -10.0 -1.0 0.0 1.0 10.0} -shape {6} -dtype float32]
    
    set result [torch::softplus $input]
    set values [torch::tensorToList $result]
    
    ;# Check that all values are positive
    set all_positive 1
    foreach val $values {
        if {$val <= 0.0} {
            set all_positive 0
            break
        }
    }
    set all_positive
} {1}

;# Test 7: Mathematical correctness - approximates max(0,x) for large positive x
test softplus-4.2 {Mathematical correctness - approximates input for large positive values} {
    set input [torch::tensorCreate -data {10.0} -shape {1} -dtype float32]
    
    set result [torch::softplus $input]
    set output_val [torch::tensorItem $result]
    
    ;# For large positive values, softplus(x) ≈ x
    ;# softplus(10) should be approximately 10.0000454
    expr {abs($output_val - 10.0) < 0.001}
} {1}

;# Test 8: Mathematical correctness - approximates log(2) for x=0
test softplus-4.3 {Mathematical correctness - softplus(0) ≈ log(2)} {
    set input [torch::tensorCreate -data {0.0} -shape {1} -dtype float32]
    
    set result [torch::softplus $input]
    set output_val [torch::tensorItem $result]
    
    ;# softplus(0) = log(1 + exp(0)) = log(2) ≈ 0.693
    expr {abs($output_val - 0.693) < 0.01}
} {1}

;# Test 9: Multi-dimensional tensors
test softplus-5.1 {Multi-dimensional tensor} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0 -2.0 2.0 3.0} -shape {2 3} -dtype float32]
    
    set result [torch::softplus -input $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {2 3}

;# Test 10: Batch processing
test softplus-5.2 {Batch processing with 3D tensor} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0 2.0 -2.0 -1.0 0.0 1.0} -shape {2 2 2} -dtype float32]
    
    set result [torch::softplus -input $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {2 2 2}

;# Test 11: Syntax consistency - both syntaxes produce same result
test softplus-6.1 {Syntax consistency} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0 2.0} -shape {4} -dtype float32]
    
    set result1 [torch::softplus $input]
    set result2 [torch::softplus -input $input]
    
    ;# Check if results are close (using element-wise comparison)
    set diff [torch::tensorSub $result1 $result2]
    set abs_diff [torch::tensorAbs $diff]
    set max_diff [torch::tensorMax $abs_diff]
    set max_val [torch::tensorItem $max_diff]
    
    expr {$max_val < 1e-6}
} {1}

;# Test 12: CamelCase alias consistency
test softplus-6.2 {CamelCase alias consistency} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0} -shape {3} -dtype float32]
    
    set result1 [torch::softplus -input $input]
    set result2 [torch::softPlus -input $input]
    
    ;# Check if results are identical
    set diff [torch::tensorSub $result1 $result2]
    set abs_diff [torch::tensorAbs $diff]
    set max_diff [torch::tensorMax $abs_diff]
    set max_val [torch::tensorItem $max_diff]
    
    expr {$max_val < 1e-6}
} {1}

;# Test 13: Error handling - no parameters
test softplus-7.1 {Error handling - no parameters} {
    set result [catch {torch::softplus} msg]
    list $result [string match "*tensor*" $msg]
} {1 1}

;# Test 14: Error handling - invalid tensor handle
test softplus-7.2 {Error handling - invalid tensor handle} {
    set result [catch {torch::softplus invalid_tensor} msg]
    list $result [string match "*Invalid tensor name*" $msg]
} {1 1}

;# Test 15: Error handling - missing input parameter in named syntax
test softplus-7.3 {Error handling - missing input parameter} {
    set result [catch {torch::softplus -unknown value} msg]
    list $result [string match "*Unknown parameter*" $msg]
} {1 1}

;# Test 16: Error handling - missing value for named parameter
test softplus-7.4 {Error handling - missing value for named parameter} {
    set result [catch {torch::softplus -input} msg]
    list $result [string match "*wrong*args*" $msg]
} {1 1}

;# Test 17: Error handling - too many positional arguments
test softplus-7.5 {Error handling - too many positional arguments} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set result [catch {torch::softplus $input extra_arg} msg]
    list $result [string match "*wrong*args*" $msg]
} {1 1}

;# Test 18: Different data types
test softplus-8.1 {Different data types - float64} {
    set input [torch::tensorCreate -data {-1.0 0.0 1.0} -shape {3} -dtype float64]
    
    set result [torch::softplus $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {3}

;# Test 19: Edge cases - very negative values
test softplus-8.2 {Edge cases - very negative values} {
    set input [torch::tensorCreate -data {-20.0 -10.0} -shape {2} -dtype float32]
    
    set result [torch::softplus $input]
    set values [torch::tensorToList $result]
    
    ;# Very negative values should produce very small positive results
    set first_val [lindex $values 0]
    set second_val [lindex $values 1]
    
    expr {$first_val > 0.0 && $first_val < 1e-8 && $second_val > 0.0 && $second_val < 1e-4}
} {1}

cleanupTests 