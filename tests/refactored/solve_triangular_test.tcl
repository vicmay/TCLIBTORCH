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

;# Helper function to create test matrices
proc setup_triangular_system {} {
    ;# Create upper triangular matrix A (3x3)
    set A [torch::tensorCreate -data {3.0 2.0 1.0 0.0 4.0 2.0 0.0 0.0 5.0} -shape {3 3} -dtype float32]
    
    ;# Create right-hand side B (3x1)  
    set B [torch::tensorCreate -data {14.0 18.0 15.0} -shape {3 1} -dtype float32]
    
    return [list $A $B]
}

proc setup_lower_triangular_system {} {
    ;# Create lower triangular matrix A (3x3)
    set A [torch::tensorCreate -data {2.0 0.0 0.0 1.0 3.0 0.0 4.0 2.0 1.0} -shape {3 3} -dtype float32]
    
    ;# Create right-hand side B (3x1)
    set B [torch::tensorCreate -data {4.0 8.0 18.0} -shape {3 1} -dtype float32]
    
    return [list $A $B]
}

;# Test 1: Basic positional syntax (upper triangular)
test solve_triangular-1.1 {Basic positional syntax - upper triangular} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [torch::solve_triangular $B $A 1 1 0]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 1}}

;# Test 2: Named parameter syntax (upper triangular)
test solve_triangular-2.1 {Named parameter syntax - upper triangular} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [torch::solve_triangular -B $B -A $A -upper 1 -left 1 -unitriangular 0]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 1}}

;# Test 3: camelCase alias
test solve_triangular-3.1 {camelCase alias - solveTriangular} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [torch::solveTriangular -B $B -A $A -upper 1 -left 1]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 1}}

;# Test 4: Lower triangular system (positional)
test solve_triangular-4.1 {Lower triangular system - positional syntax} {
    set system [setup_lower_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [torch::solve_triangular $B $A 0 1 0]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 1}}

;# Test 5: Lower triangular system (named parameters)  
test solve_triangular-5.1 {Lower triangular system - named parameters} {
    set system [setup_lower_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [torch::solve_triangular -B $B -A $A -upper 0 -left 1 -unitriangular 0]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 1}}

;# Test 6: Default parameter values (named syntax)
test solve_triangular-6.1 {Default parameter values - only required params} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    ;# Should use default values: upper=true, left=true, unitriangular=false
    set result [torch::solve_triangular -B $B -A $A]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 1}}

;# Test 7: Mixed case parameter names
test solve_triangular-7.1 {Mixed case parameter names} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [torch::solve_triangular -b $B -a $A -upper 1]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 1}}

;# Test 8: Batch processing (multiple right-hand sides)
test solve_triangular-8.1 {Batch processing with multiple RHS} {
    ;# Create upper triangular matrix A (3x3)
    set A [torch::tensorCreate -data {3.0 2.0 1.0 0.0 4.0 2.0 0.0 0.0 5.0} -shape {3 3} -dtype float32]
    
    ;# Create batch B (3x2) - two different right-hand sides
    set B [torch::tensorCreate -data {14.0 21.0 18.0 26.0 15.0 20.0} -shape {3 2} -dtype float32]
    
    set result [torch::solve_triangular -B $B -A $A -upper 1]
    set shape [torch::tensorShape $result]
    
    list $shape
} {{3 2}}

;# Test 9: Syntax consistency - both syntaxes produce same result
test solve_triangular-9.1 {Syntax consistency} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result1 [torch::solve_triangular $B $A 1 1 0]
    set result2 [torch::solve_triangular -B $B -A $A -upper 1 -left 1 -unitriangular 0]
    
    ;# Check if results are close (using tensor_norm with tolerance)
    set diff [torch::tensorSub $result1 $result2]
    set norm [torch::tensor_norm $diff]
    set norm_val [torch::tensorItem $norm]
    
    expr {$norm_val < 1e-6}
} {1}

;# Test 10: Error handling - missing required parameters
test solve_triangular-10.1 {Error handling - missing B parameter} {
    set A [torch::tensorCreate -data {1.0 2.0 0.0 3.0} -shape {2 2} -dtype float32]
    
    set result [catch {torch::solve_triangular -A $A} msg]
    list $result [string match "*Required parameters missing*" $msg]
} {1 1}

;# Test 11: Error handling - missing A parameter  
test solve_triangular-10.2 {Error handling - missing A parameter} {
    set B [torch::tensorCreate -data {1.0 2.0} -shape {2 1} -dtype float32]
    
    set result [catch {torch::solve_triangular -B $B} msg]
    list $result [string match "*Required parameters missing*" $msg]
} {1 1}

;# Test 12: Error handling - invalid tensor handle
test solve_triangular-10.3 {Error handling - invalid tensor handle} {
    set B [torch::tensorCreate -data {1.0 2.0} -shape {2 1} -dtype float32]
    
    set result [catch {torch::solve_triangular -B $B -A "invalid_handle"} msg]
    list $result [string match "*Invalid A tensor*" $msg]
} {1 1}

;# Test 13: Error handling - unknown parameter
test solve_triangular-10.4 {Error handling - unknown parameter} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [catch {torch::solve_triangular -B $B -A $A -unknown_param 1} msg]
    list $result [string match "*Unknown parameter*" $msg]
} {1 1}

;# Test 14: Error handling - missing value for parameter
test solve_triangular-10.5 {Error handling - missing value for parameter} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [catch {torch::solve_triangular -B $B -A $A -upper} msg]
    list $result [string match "*Missing value for parameter*" $msg]
} {1 1}

;# Test 15: Error handling - invalid boolean parameter
test solve_triangular-10.6 {Error handling - invalid boolean parameter} {
    set system [setup_triangular_system]
    set A [lindex $system 0]
    set B [lindex $system 1]
    
    set result [catch {torch::solve_triangular -B $B -A $A -upper "invalid"} msg]
    list $result [string match "*Invalid upper parameter*" $msg]
} {1 1}

cleanupTests 