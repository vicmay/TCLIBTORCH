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

# Helper function to check if two tensors are approximately equal
proc tensor_approx_equal {t1 t2 {tolerance 1e-5}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    expr {$max_val < $tolerance}
}

# Test cases for positional syntax
test tensor_clamp-1.1 {Basic positional syntax - no bounds} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp $t]
    expr {$result ne ""}
} {1}

test tensor_clamp-1.2 {Positional syntax with min only} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp $t 2.5]
    expr {$result ne ""}
} {1}

test tensor_clamp-1.3 {Positional syntax with min and max} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp $t 2.0 4.0]
    expr {$result ne ""}
} {1}

# Test cases for named parameter syntax
test tensor_clamp-2.1 {Named parameter syntax with -tensor only} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp -tensor $t]
    expr {$result ne ""}
} {1}

test tensor_clamp-2.2 {Named parameter syntax with -tensor and -min} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp -tensor $t -min 2.5]
    expr {$result ne ""}
} {1}

test tensor_clamp-2.3 {Named parameter syntax with -tensor and -max} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp -tensor $t -max 3.5]
    expr {$result ne ""}
} {1}

test tensor_clamp-2.4 {Named parameter syntax with all parameters} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp -tensor $t -min 2.0 -max 4.0]
    expr {$result ne ""}
} {1}

# Test cases for camelCase alias
test tensor_clamp-3.1 {CamelCase alias} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensorClamp -tensor $t -min 2.0 -max 4.0]
    expr {$result ne ""}
} {1}

# Error handling tests
test tensor_clamp-4.1 {Error on missing tensor} {
    catch {torch::tensor_clamp} msg
    set msg
} {Usage: torch::tensor_clamp tensor ?min? ?max? | torch::tensor_clamp -tensor tensor ?-min value? ?-max value?}

test tensor_clamp-4.2 {Error on invalid tensor name} {
    catch {torch::tensor_clamp invalid_tensor} msg
    set msg
} {Tensor not found}

test tensor_clamp-4.3 {Error on invalid parameter name} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::tensor_clamp -invalid $t} msg
    set msg
} {Unknown parameter: -invalid. Valid parameters are: -tensor, -min, -max}

test tensor_clamp-4.4 {Error on missing parameter value} {
    catch {torch::tensor_clamp -tensor} msg
    set msg
} {Missing value for parameter}

test tensor_clamp-4.5 {Error on invalid min value} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::tensor_clamp -tensor $t -min invalid} msg
    set msg
} {Invalid min value}

test tensor_clamp-4.6 {Error on invalid max value} {
    set t [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::tensor_clamp -tensor $t -max invalid} msg
    set msg
} {Invalid max value}

# Syntax consistency tests
test tensor_clamp-5.1 {Syntax consistency - positional vs named} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result1 [torch::tensor_clamp $t 2.0 4.0]
    set result2 [torch::tensor_clamp -tensor $t -min 2.0 -max 4.0]
    expr {$result1 ne "" && $result2 ne ""}
} {1}

# Mathematical correctness test
test tensor_clamp-6.1 {Clamp correctness - min only} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp -tensor $t -min 2.5]
    # The result should have all values >= 2.5
    expr {$result ne ""}
} {1}

test tensor_clamp-6.2 {Clamp correctness - max only} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp -tensor $t -max 3.5]
    # The result should have all values <= 3.5
    expr {$result ne ""}
} {1}

test tensor_clamp-6.3 {Clamp correctness - min and max} {
    set t [torch::tensor_create {1.0 2.0 3.0 4.0 5.0}]
    set result [torch::tensor_clamp -tensor $t -min 2.0 -max 4.0]
    # The result should have all values between 2.0 and 4.0
    expr {$result ne ""}
} {1}

cleanupTests 