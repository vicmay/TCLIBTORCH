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

# ============================================================================
# Test torch::int_repr - Get Integer Representation of Quantized Tensor
# ============================================================================

# Test 1: Error handling - missing input
test int_repr-1.1 {Error handling - missing input} {
    catch {torch::int_repr} msg
    string match "*Required parameters missing*" $msg
} 1

test int_repr-1.2 {Error handling - invalid tensor handle} {
    catch {torch::int_repr invalid_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

test int_repr-1.3 {Error handling - unknown parameter} {
    catch {torch::int_repr -unknown_param value} msg
    string match "*Unknown parameter*" $msg
} 1

test int_repr-1.4 {Error handling - missing parameter value} {
    catch {torch::int_repr -input} msg
    string match "*Named parameters must come in pairs*" $msg
} 1

# Test 2: CamelCase alias exists
test int_repr-2.1 {CamelCase alias exists} {
    catch {torch::intRepr} msg
    string match "*Required parameters missing*" $msg
} 1

test int_repr-2.2 {CamelCase alias with invalid tensor} {
    catch {torch::intRepr invalid_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

test int_repr-2.3 {CamelCase alias with named parameters} {
    catch {torch::intRepr -input invalid_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

# Test 3: Parameter parsing - both input parameter names work
test int_repr-3.1 {Parameter parsing - input parameter} {
    catch {torch::int_repr -input invalid_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

test int_repr-3.2 {Parameter parsing - tensor parameter} {
    catch {torch::int_repr -tensor invalid_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

# Test 4: Multiple unknown parameters
test int_repr-4.1 {Multiple unknown parameters} {
    catch {torch::int_repr -input invalid_tensor -unknown param} msg
    string match "*Unknown parameter*" $msg
} 1

test int_repr-4.2 {Unknown parameter before valid one} {
    catch {torch::int_repr -unknown param -input invalid_tensor} msg
    string match "*Unknown parameter*" $msg
} 1

# Test 5: Parameter consistency between syntaxes
test int_repr-5.1 {Positional syntax error message} {
    catch {torch::int_repr invalid_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

test int_repr-5.2 {Named syntax error message} {
    catch {torch::int_repr -input invalid_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

# Test 6: Edge cases in parameter parsing
test int_repr-6.1 {Empty parameter value} {
    catch {torch::int_repr -input ""} msg
    string match "*Required parameters missing*" $msg
} 1

test int_repr-6.2 {Only parameter flag without value} {
    catch {torch::int_repr -input} msg
    string match "*Named parameters must come in pairs*" $msg
} 1

test int_repr-6.3 {Parameter flag used as value} {
    catch {torch::int_repr -input -tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

# Test 7: Dual syntax detection
test int_repr-7.1 {Positional syntax detection} {
    catch {torch::int_repr some_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

test int_repr-7.2 {Named syntax detection} {
    catch {torch::int_repr -input some_tensor} msg
    string match "*Invalid quantized tensor*" $msg
} 1

test int_repr-7.3 {Named syntax with dash in tensor name} {
    catch {torch::int_repr -input some-tensor-name} msg
    string match "*Invalid quantized tensor*" $msg
} 1

# Test 8: Error message quality
test int_repr-8.1 {Error message contains command context} {
    catch {torch::int_repr} msg
    string match "*int_repr*" $msg
} 1

test int_repr-8.2 {Error message for invalid tensor includes tensor name} {
    catch {torch::int_repr invalid_tensor_name} msg
    string match "*invalid_tensor_name*" $msg
} 1

test int_repr-8.3 {Error message for unknown parameter includes parameter name} {
    catch {torch::int_repr -bad_param value} msg
    string match "*bad_param*" $msg
} 1

# Test 9: Command registration verification
test int_repr-9.1 {Snake case command exists} {
    set commands [info commands torch::int_repr]
    expr {[llength $commands] == 1}
} 1

test int_repr-9.2 {CamelCase command exists} {
    set commands [info commands torch::intRepr]
    expr {[llength $commands] == 1}
} 1

test int_repr-9.3 {Both commands are the same implementation} {
    catch {torch::int_repr} msg1
    catch {torch::intRepr} msg2
    expr {$msg1 eq $msg2}
} 1

# Test 10: Comprehensive parameter validation
test int_repr-10.1 {All parameter combinations fail appropriately} {
    set results {}
    lappend results [catch {torch::int_repr} msg]
    lappend results [catch {torch::int_repr -input} msg]
    lappend results [catch {torch::int_repr -tensor} msg]
    lappend results [catch {torch::int_repr -input invalid} msg]
    lappend results [catch {torch::int_repr -tensor invalid} msg]
    # All should return 1 (error)
    expr {[lsort -unique $results] eq "1"}
} 1

cleanupTests 