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

;# Test 1: Basic positional syntax (default dimension)
test softmin-1.1 {Basic positional syntax - default dimension} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]
    
    set result [torch::softmin $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {4}

;# Test 2: Positional syntax with explicit dimension
test softmin-1.2 {Positional syntax with explicit dimension} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    
    set result [torch::softmin $input 1]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{2 3}}

;# Test 3: Named parameter syntax - basic
test softmin-2.1 {Named parameter syntax - basic} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]
    
    set result [torch::softmin -input $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {4}

;# Test 4: Named parameter syntax with dimension
test softmin-2.2 {Named parameter syntax with dimension} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    
    set result [torch::softmin -input $input -dim 0]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{2 3}}

;# Test 5: Alternative parameter names
test softmin-2.3 {Named syntax with -tensor parameter} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set result [torch::softmin -tensor $input -dimension 0]
    set shape [torch::tensorShape $result]
    
    set shape
} {3}

;# Test 6: camelCase alias - basic
test softmin-3.1 {camelCase alias - positional syntax} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]
    
    set result [torch::softMin $input]
    set shape [torch::tensorShape $result]
    
    set shape
} {4}

;# Test 7: camelCase alias - named syntax
test softmin-3.2 {camelCase alias - named syntax} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    
    set result [torch::softMin -input $input -dim 1]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{2 3}}

;# Test 8: Mathematical correctness - softmin properties
test softmin-4.1 {Mathematical correctness - sum to 1} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set result [torch::softmin $input]
    set sum [torch::tensorSum $result]
    set sum_val [torch::tensorItem $sum]
    
    ;# Check if sum is approximately 1.0 (within tolerance)
    expr {abs($sum_val - 1.0) < 1e-6}
} {1}

;# Test 9: Mathematical correctness - smaller values get higher probabilities
test softmin-4.2 {Mathematical correctness - inverse of softmax} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set softmin_result [torch::softmin $input]
    set softmin_values [torch::tensorToList $softmin_result]
    
    ;# First element should have highest probability (smallest input)
    set first [lindex $softmin_values 0]
    set second [lindex $softmin_values 1]
    set third [lindex $softmin_values 2]
    
    expr {$first > $second && $second > $third}
} {1}

;# Test 10: Multi-dimensional tensor with different dimensions
test softmin-5.1 {Multi-dimensional tensor - dim 0} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    
    set result [torch::softmin -input $input -dim 0]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{2 3}}

;# Test 11: Multi-dimensional tensor - dim 1
test softmin-5.2 {Multi-dimensional tensor - dim 1} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    
    set result [torch::softmin -input $input -dim 1]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{2 3}}

;# Test 12: Syntax consistency - both syntaxes produce same result
test softmin-6.1 {Syntax consistency} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]
    
    set result1 [torch::softmin $input 0]
    set result2 [torch::softmin -input $input -dim 0]
    
    ;# Check if results are close (using element-wise comparison)
    set diff [torch::tensorSub $result1 $result2]
    set abs_diff [torch::tensorAbs $diff]
    set max_diff [torch::tensorMax $abs_diff]
    set max_val [torch::tensorItem $max_diff]
    
    expr {$max_val < 1e-6}
} {1}

;# Test 13: Error handling - missing required parameters
test softmin-7.1 {Error handling - no parameters} {
    set result [catch {torch::softmin} msg]
    list $result [string match "*Usage*" $msg]
} {1 1}

;# Test 14: Error handling - invalid tensor handle
test softmin-7.2 {Error handling - invalid tensor handle} {
    set result [catch {torch::softmin invalid_tensor} msg]
    list $result [string match "*Invalid tensor name*" $msg]
} {1 1}

;# Test 15: Error handling - invalid dimension
test softmin-7.3 {Error handling - invalid dimension in positional syntax} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set result [catch {torch::softmin $input abc} msg]
    list $result [string match "*Invalid dimension*" $msg]
} {1 1}

;# Test 16: Error handling - invalid dimension in named syntax
test softmin-7.4 {Error handling - invalid dimension in named syntax} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set result [catch {torch::softmin -input $input -dim abc} msg]
    list $result [string match "*Invalid dimension*" $msg]
} {1 1}

;# Test 17: Error handling - unknown parameter
test softmin-7.5 {Error handling - unknown parameter} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set result [catch {torch::softmin -input $input -unknown param} msg]
    list $result [string match "*Unknown parameter*" $msg]
} {1 1}

;# Test 18: Error handling - missing parameter value
test softmin-7.6 {Error handling - missing parameter value} {
    set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
    
    set result [catch {torch::softmin -input $input -dim} msg]
    list $result [string match "*Missing value*" $msg]
} {1 1}

cleanupTests 