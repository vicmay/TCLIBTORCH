#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax

test benchmark-1.1 {positional matmul basic} {
    set result [torch::benchmark matmul]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-1.2 {positional with iterations} {
    set result [torch::benchmark matmul 2]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-1.3 {positional with size} {
    set result [torch::benchmark matmul 1 500x500]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-1.4 {positional with dtype} {
    set result [torch::benchmark matmul 1 100x100 float64]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-1.5 {positional add operation} {
    set result [torch::benchmark add 1 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

# Test 2: Named parameter syntax

test benchmark-2.1 {named basic operation} {
    set result [torch::benchmark -operation matmul]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-2.2 {named with iterations} {
    set result [torch::benchmark -operation matmul -iterations 2]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-2.3 {named with size} {
    set result [torch::benchmark -operation matmul -size 200x200]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-2.4 {named with dtype} {
    set result [torch::benchmark -operation matmul -dtype float64 -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-2.5 {named add operation} {
    set result [torch::benchmark -operation add -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-2.6 {named with verbose} {
    set result [torch::benchmark -operation matmul -verbose 1 -size 100x100]
    string match "*Operation: matmul*" $result
} {1}

test benchmark-2.7 {named parameter aliases} {
    set result [torch::benchmark -op matmul -iter 1 -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

# Test 3: Different operations

test benchmark-3.1 {matmul operation} {
    set result [torch::benchmark -operation matmul -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-3.2 {mm operation alias} {
    set result [torch::benchmark -operation mm -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-3.3 {add operation} {
    set result [torch::benchmark -operation add -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

# Test 4: Error handling

test benchmark-4.1 {missing operation in named syntax} {
    catch {torch::benchmark -iterations 1} result
    string match "*Missing required parameter: -operation*" $result
} {1}

test benchmark-4.2 {unknown parameter} {
    catch {torch::benchmark -unknown value} result
    string match "*Unknown parameter: -unknown*" $result
} {1}

test benchmark-4.3 {invalid iterations} {
    catch {torch::benchmark -operation matmul -iterations -1} result
    string match "*Invalid iterations: must be positive integer*" $result
} {1}

test benchmark-4.4 {unknown operation} {
    catch {torch::benchmark -operation unknown_op} result
    string match "*Unknown operation: unknown_op*" $result
} {1}

test benchmark-4.5 {invalid verbose value} {
    catch {torch::benchmark -operation matmul -verbose invalid} result
    string match "*Invalid verbose*" $result
} {1}

# Test 5: Data types

test benchmark-5.1 {float32 dtype} {
    set result [torch::benchmark -operation matmul -dtype float32 -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-5.2 {float64 dtype} {
    set result [torch::benchmark -operation matmul -dtype float64 -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-5.3 {double dtype} {
    set result [torch::benchmark -operation matmul -dtype double -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

# Test 6: Device handling

test benchmark-6.1 {cpu device} {
    set result [torch::benchmark -operation matmul -device cpu -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

# Test 7: Size variations

test benchmark-7.1 {square matrix} {
    set result [torch::benchmark -operation matmul -size 200x200]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-7.2 {single dimension} {
    set result [torch::benchmark -operation add -size 1000]
    expr {[string is integer $result] && $result > 0}
} {1}

# Test 8: Verbose output

test benchmark-8.1 {verbose disabled} {
    set result [torch::benchmark -operation matmul -verbose 0 -size 100x100]
    expr {[string is integer $result] && $result > 0}
} {1}

test benchmark-8.2 {verbose enabled} {
    set result [torch::benchmark -operation matmul -verbose 1 -size 100x100]
    expr {[string match "*Operation: matmul*" $result] && [string match "*Time:*" $result]}
} {1}

# Test 9: Performance characteristics

test benchmark-9.1 {multiple iterations take longer} {
    set time1 [torch::benchmark -operation matmul -iterations 1 -size 100x100]
    set time2 [torch::benchmark -operation matmul -iterations 3 -size 100x100]
    expr {$time2 > $time1}
} {1}

test benchmark-9.2 {larger size takes longer} {
    set time1 [torch::benchmark -operation matmul -size 50x50]
    set time2 [torch::benchmark -operation matmul -size 200x200]
    expr {$time2 > $time1}
} {1}

# Test 10: Equivalence of syntaxes

test benchmark-10.1 {positional vs named equivalence} {
    # Both should work and return positive times
    set pos_result [torch::benchmark matmul 1 100x100 float32]
    set named_result [torch::benchmark -operation matmul -iterations 1 -size 100x100 -dtype float32]
    expr {[string is integer $pos_result] && [string is integer $named_result] && $pos_result > 0 && $named_result > 0}
} {1}

test benchmark-10.2 {operation alias equivalence} {
    set result1 [torch::benchmark -operation matmul -size 100x100]
    set result2 [torch::benchmark -operation mm -size 100x100]
    expr {[string is integer $result1] && [string is integer $result2] && $result1 > 0 && $result2 > 0}
} {1}

cleanupTests 