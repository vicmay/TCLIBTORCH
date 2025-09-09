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

# Helper function to check tensor values
proc check_tensor_values {tensor expected_values {tolerance 1e-5}} {
    set values [torch::tensorToList $tensor]
    if {[llength $values] != [llength $expected_values]} {
        return "Length mismatch: got [llength $values], expected [llength $expected_values]"
    }
    
    foreach v $values e $expected_values {
        if {abs($v - $e) > $tolerance} {
            return "Value mismatch: got $v, expected $e (tolerance: $tolerance)"
        }
    }
    return ""
}

# Test cases for positional syntax
test softsign-1.1 {Basic softsign with positional syntax} {
    set t [torch::tensorCreate -data {1.0 -1.0 2.0 -2.0} -shape {4} -dtype float32]
    set result [torch::softsign $t]
    set expected {0.5 -0.5 0.6666666666666666 -0.6666666666666666}
    check_tensor_values $result $expected
} {}

test softsign-1.2 {Softsign with zero values} {
    set t [torch::tensorCreate -data {0.0 0.0 0.0} -shape {3} -dtype float32]
    set result [torch::softsign $t]
    set expected {0.0 0.0 0.0}
    check_tensor_values $result $expected
} {}

test softsign-1.3 {Softsign with large values} {
    set t [torch::tensorCreate -data {100.0 -100.0} -shape {2} -dtype float32]
    set result [torch::softsign $t]
    set expected {0.99009901 -0.99009901}
    check_tensor_values $result $expected
} {}

test softsign-1.4 {Softsign with 2D tensor} {
    set t [torch::tensorCreate -data {1.0 -1.0 2.0 -2.0} -shape {2 2} -dtype float32]
    set result [torch::softsign $t]
    set expected {0.5 -0.5 0.6666666666666666 -0.6666666666666666}
    check_tensor_values $result $expected
} {}

# Test cases for named parameter syntax
test softsign-2.1 {Basic softsign with named parameters} {
    set t [torch::tensorCreate -data {1.0 -1.0 2.0 -2.0} -shape {4} -dtype float32]
    set result [torch::softsign -input $t]
    set expected {0.5 -0.5 0.6666666666666666 -0.6666666666666666}
    check_tensor_values $result $expected
} {}

test softsign-2.2 {Softsign with -tensor alias} {
    set t [torch::tensorCreate -data {1.0 -1.0 2.0 -2.0} -shape {4} -dtype float32]
    set result [torch::softsign -tensor $t]
    set expected {0.5 -0.5 0.6666666666666666 -0.6666666666666666}
    check_tensor_values $result $expected
} {}

# Test cases for camelCase alias
test softsign-3.1 {Basic softsign with camelCase alias - positional} {
    set t [torch::tensorCreate -data {1.0 -1.0 2.0 -2.0} -shape {4} -dtype float32]
    set result [torch::softSign $t]
    set expected {0.5 -0.5 0.6666666666666666 -0.6666666666666666}
    check_tensor_values $result $expected
} {}

test softsign-3.2 {Softsign with camelCase alias - named parameters} {
    set t [torch::tensorCreate -data {1.0 -1.0 2.0 -2.0} -shape {4} -dtype float32]
    set result [torch::softSign -input $t]
    set expected {0.5 -0.5 0.6666666666666666 -0.6666666666666666}
    check_tensor_values $result $expected
} {}

# Error handling tests
test softsign-4.1 {Error on missing tensor} {
    catch {torch::softsign} result
    set result
} {wrong # args: should be "torch::softsign tensor | -input tensor"}

test softsign-4.2 {Error on invalid tensor name} {
    catch {torch::softsign invalid_tensor} result
    set result
} {Invalid tensor name}

test softsign-4.3 {Error on invalid named parameter} {
    set t [torch::tensorCreate -data {1.0 -1.0} -shape {2} -dtype float32]
    catch {torch::softsign -invalid $t} result
    set result
} {Unknown parameter: -invalid}

test softsign-4.4 {Error on missing value for named parameter} {
    set t [torch::tensorCreate -data {1.0 -1.0} -shape {2} -dtype float32]
    catch {torch::softsign -input} result
    set result
} {wrong # args: should be "torch::softsign tensor | -input tensor"}

# Mathematical property tests
test softsign-5.1 {Softsign is bounded between -1 and 1} {
    set t [torch::tensorCreate -data {1000.0 -1000.0 10000.0 -10000.0} -shape {4} -dtype float32]
    set result [torch::softsign $t]
    set values [torch::tensorToList $result]
    set all_bounded 1
    foreach v $values {
        if {$v <= -1.0 || $v >= 1.0} {
            set all_bounded 0
            break
        }
    }
    set all_bounded
} {1}

test softsign-5.2 {Softsign preserves sign} {
    set t [torch::tensorCreate -data {1.0 -1.0 2.0 -2.0 0.0} -shape {5} -dtype float32]
    set result [torch::softsign $t]
    set input_values [torch::tensorToList $t]
    set output_values [torch::tensorToList $result]
    set signs_preserved 1
    foreach in $input_values out $output_values {
        if {($in > 0 && $out <= 0) || ($in < 0 && $out >= 0) || ($in == 0 && $out != 0)} {
            set signs_preserved 0
            break
        }
    }
    set signs_preserved
} {1}

test softsign-5.3 {Softsign is continuous at zero} {
    set t [torch::tensorCreate -data {-0.001 0.0 0.001} -shape {3} -dtype float32]
    set result [torch::softsign $t]
    set expected {-0.000999 0.0 0.000999}
    check_tensor_values $result $expected 1e-6
} {}

cleanupTests 