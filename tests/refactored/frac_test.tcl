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

# ============================
# Test Cases for torch::frac
# ============================

# Test 1: Basic Positional Syntax
test frac-1.1 {Basic positional syntax} {
    set input [torch::tensor_create -data {2.3 -1.7 3.5 -0.5 0.8} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 5}
} 1

test frac-1.2 {Positional syntax with multi-dimensional tensor} {
    set input [torch::tensor_create -data {1.2 -2.3 3.4 -4.5} -shape {2 2} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-1.3 {Positional syntax with zero values} {
    set input [torch::tensor_create -data {0.0 -0.0 0.5 -0.5} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-1.4 {Positional syntax with single value} {
    set input [torch::tensor_create -data {4.7} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 1}
} 1

# Test 2: Named Parameter Syntax
test frac-2.1 {Named parameter syntax with -input} {
    set input [torch::tensor_create -data {3.2 -1.8 2.5 -0.3} -dtype float32]
    set result [torch::frac -input $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-2.2 {Named parameter syntax with -tensor} {
    set input [torch::tensor_create -data {1.5 -2.5 3.5 -4.5} -dtype float32]
    set result [torch::frac -tensor $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-2.3 {Named parameter syntax with complex tensor} {
    set input [torch::tensor_create -data {1.1 2.2 3.3 4.4 5.5 6.6} -shape {2 3} -dtype float32]
    set result [torch::frac -input $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 6}
} 1

test frac-2.4 {Named parameter syntax with negative values} {
    set input [torch::tensor_create -data {-1.3 -2.7 -3.9 -4.1} -dtype float32]
    set result [torch::frac -input $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-2.5 {Named parameter syntax with mixed values} {
    set input [torch::tensor_create -data {1.2 -1.2 0.8 -0.8 2.5 -2.5} -dtype float32]
    set result [torch::frac -input $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 6}
} 1

# Test 3: CamelCase Alias
test frac-3.1 {CamelCase alias basic} {
    set input [torch::tensor_create -data {2.3 -1.7 3.5 -0.5} -dtype float32]
    set result [torch::Frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-3.2 {CamelCase alias with named params} {
    set input [torch::tensor_create -data {4.2 -3.8 5.1 -2.9} -dtype float32]
    set result [torch::Frac -input $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-3.3 {CamelCase alias with complex tensor} {
    set input [torch::tensor_create -data {1.1 2.2 3.3 4.4} -shape {2 2} -dtype float32]
    set result [torch::Frac -input $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-3.4 {CamelCase alias syntax consistency} {
    set input [torch::tensor_create -data {2.5} -dtype float32]
    set r1 [torch::frac $input]
    set r2 [torch::frac -input $input]
    set r3 [torch::Frac -input $input]
    set v1 [torch::tensor_item $r1]
    set v2 [torch::tensor_item $r2]
    set v3 [torch::tensor_item $r3]
    expr {abs($v1 - $v2) < 1e-6 && abs($v2 - $v3) < 1e-6}
} 1

# Test 4: Error Handling
test frac-4.1 {Error: missing arguments} {
    set code [catch {torch::frac} msg]
    expr {$code == 1}
} 1

test frac-4.2 {Error: invalid tensor name} {
    set code [catch {torch::frac invalid_tensor} msg]
    expr {$code == 1}
} 1

test frac-4.3 {Error: missing named parameter value} {
    set code [catch {torch::frac -input} msg]
    expr {$code == 1}
} 1

test frac-4.4 {Error: unknown named parameter} {
    set input [torch::tensor_create -data {1.5 2.5} -dtype float32]
    set code [catch {torch::frac -invalid_param $input} msg]
    expr {$code == 1}
} 1

test frac-4.5 {Error: too many positional arguments} {
    set input [torch::tensor_create -data {1.5 2.5} -dtype float32]
    set code [catch {torch::frac $input extra_arg} msg]
    expr {$code == 1}
} 1

test frac-4.6 {Error: empty tensor name} {
    set code [catch {torch::frac ""} msg]
    expr {$code == 1}
} 1

test frac-4.7 {Error: named parameter with empty value} {
    set code [catch {torch::frac -input ""} msg]
    expr {$code == 1}
} 1

test frac-4.8 {Error: camelCase with wrong parameter} {
    set input [torch::tensor_create -data {1.5 2.5} -dtype float32]
    set code [catch {torch::Frac -wrong_param $input} msg]
    expr {$code == 1}
} 1

test frac-4.9 {Error: parameter without value at end} {
    set input [torch::tensor_create -data {1.5 2.5} -dtype float32]
    set code [catch {torch::frac -input $input -tensor} msg]
    expr {$code == 1}
} 1

# Test 5: Edge Cases
test frac-5.1 {Edge case: very small positive values} {
    set input [torch::tensor_create -data {0.001 0.0001 0.00001} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 3}
} 1

test frac-5.2 {Edge case: very small negative values} {
    set input [torch::tensor_create -data {-0.001 -0.0001 -0.00001} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 3}
} 1

test frac-5.3 {Edge case: large positive values} {
    set input [torch::tensor_create -data {1000.5 2000.3 3000.7} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 3}
} 1

test frac-5.4 {Edge case: large negative values} {
    set input [torch::tensor_create -data {-1000.5 -2000.3 -3000.7} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 3}
} 1

test frac-5.5 {Edge case: values close to integers} {
    set input [torch::tensor_create -data {2.999999 3.000001 -2.999999 -3.000001} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-5.6 {Edge case: exact integers} {
    set input [torch::tensor_create -data {1.0 2.0 -3.0 -4.0} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

# Test 6: Mathematical Correctness
test frac-6.1 {Mathematical correctness: positive fractional value} {
    set input [torch::tensor_create -data {2.3} -dtype float32]
    set result [torch::frac $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.3) < 1e-5}
} 1

test frac-6.2 {Mathematical correctness: negative fractional value} {
    set input [torch::tensor_create -data {-2.3} -dtype float32]
    set result [torch::frac $input]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.3)) < 1e-5}
} 1

test frac-6.3 {Mathematical correctness: zero value} {
    set input [torch::tensor_create -data {0.0} -dtype float32]
    set result [torch::frac $input]
    set value [torch::tensor_item $result]
    expr {abs($value) < 1e-6}
} 1

test frac-6.4 {Mathematical correctness: integer value} {
    set input [torch::tensor_create -data {2.0} -dtype float32]
    set result [torch::frac $input]
    set value [torch::tensor_item $result]
    expr {abs($value) < 1e-6}
} 1

test frac-6.5 {Mathematical correctness: small positive fraction} {
    set input [torch::tensor_create -data {0.7} -dtype float32]
    set result [torch::frac $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.7) < 1e-6}
} 1

test frac-6.6 {Mathematical correctness: small negative fraction} {
    set input [torch::tensor_create -data {-0.7} -dtype float32]
    set result [torch::frac $input]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.7)) < 1e-6}
} 1

# Test 7: Syntax Consistency
test frac-7.1 {Syntax consistency: positional vs named} {
    set input [torch::tensor_create -data {2.5} -dtype float32]
    set r1 [torch::frac $input]
    set r2 [torch::frac -input $input]
    set v1 [torch::tensor_item $r1]
    set v2 [torch::tensor_item $r2]
    expr {abs($v1 - $v2) < 1e-6}
} 1

test frac-7.2 {Syntax consistency: different named parameters} {
    set input [torch::tensor_create -data {1.3} -dtype float32]
    set r1 [torch::frac -input $input]
    set r2 [torch::frac -tensor $input]
    set v1 [torch::tensor_item $r1]
    set v2 [torch::tensor_item $r2]
    expr {abs($v1 - $v2) < 1e-6}
} 1

test frac-7.3 {Syntax consistency: all three syntaxes} {
    set input [torch::tensor_create -data {4.2} -dtype float32]
    set r1 [torch::frac $input]
    set r2 [torch::frac -input $input]
    set r3 [torch::Frac -input $input]
    set v1 [torch::tensor_item $r1]
    set v2 [torch::tensor_item $r2]
    set v3 [torch::tensor_item $r3]
    expr {abs($v1 - $v2) < 1e-6 && abs($v2 - $v3) < 1e-6}
} 1

# Test 8: Integration Tests
test frac-8.1 {Integration with tensor creation} {
    set input [torch::tensor_create -data {1.5 2.5 3.5 4.5} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

test frac-8.2 {Integration with other operations} {
    set input [torch::tensor_create -data {2.3} -dtype float32]
    set frac_result [torch::frac $input]
    torch::tensor_print $frac_result
    set numel [torch::tensor_numel $frac_result]
    expr {$numel == 1}
} 1

test frac-8.3 {Integration chain validation} {
    set input [torch::tensor_create -data {1.2 2.3 3.4 4.5} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 4}
} 1

# Test 9: Performance Tests
test frac-9.1 {Performance: large tensor} {
    set large_data {}
    for {set i 0} {$i < 1000} {incr i} {
        lappend large_data [expr {$i * 0.1}]
    }
    set input [torch::tensor_create -data $large_data -dtype float32]
    set start_time [clock milliseconds]
    set result [torch::frac $input]
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    expr {$duration < 1000}
} 1

test frac-9.2 {Performance: repeated operations} {
    set input [torch::tensor_create -data {1.5 2.5 3.5 4.5} -dtype float32]
    set start_time [clock milliseconds]
    for {set i 0} {$i < 100} {incr i} {
        set result [torch::frac $input]
    }
    set end_time [clock milliseconds]
    set duration [expr {$end_time - $start_time}]
    expr {$duration < 1000}
} 1

# Test 10: Parameter Validation
test frac-10.1 {Parameter validation: basic functionality} {
    set input [torch::tensor_create -data {1.5 2.5} -dtype float32]
    set result [torch::frac $input]
    set numel [torch::tensor_numel $result]
    expr {$numel == 2}
} 1

test frac-10.2 {Parameter validation: case sensitivity} {
    set input [torch::tensor_create -data {1.5 2.5} -dtype float32]
    set code [catch {torch::frac -INPUT $input} msg]
    expr {$code == 1}
} 1

test frac-10.3 {Parameter validation: parameter alternatives} {
    set input [torch::tensor_create -data {1.5} -dtype float32]
    set r1 [torch::frac -input $input]
    set r2 [torch::frac -tensor $input]
    set v1 [torch::tensor_item $r1]
    set v2 [torch::tensor_item $r2]
    expr {abs($v1 - $v2) < 1e-6}
} 1

test frac-10.4 {Parameter validation: tensor name validation} {
    set code [catch {torch::frac -input "nonexistent_tensor"} msg]
    expr {$code == 1}
} 1

test frac-10.5 {Parameter validation: empty parameter handling} {
    set code [catch {torch::frac -input} msg]
    expr {$code == 1}
} 1

test frac-10.6 {Parameter validation: numeric tensor names} {
    set code [catch {torch::frac -input 12345} msg]
    expr {$code == 1}
} 1

# Test 11: Final Comprehensive Test
test frac-11.1 {Final comprehensive validation} {
    # Test all three syntaxes with various inputs
    set input1 [torch::tensor_create -data {1.7} -dtype float32]
    set input2 [torch::tensor_create -data {2.1} -dtype float32]
    
    # Test positional syntax
    set r1 [torch::frac $input1]
    # Test named syntax
    set r2 [torch::frac -input $input2]
    # Test camelCase syntax
    set r3 [torch::Frac -tensor $input1]
    
    set n1 [torch::tensor_numel $r1]
    set n2 [torch::tensor_numel $r2]
    set n3 [torch::tensor_numel $r3]
    
    # Verify all results have correct dimensions
    expr {$n1 == 1 && $n2 == 1 && $n3 == 1}
} 1

cleanupTests 