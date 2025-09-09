#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Create test tensors
set a [torch::tensor_create {1 2 3 4} float32]
set b [torch::tensor_create {0 2 4 4} float32]

# -----------------------------------------------------------------------------
# Test 1: Basic functionality - Positional syntax (backward compatibility)
# -----------------------------------------------------------------------------

test gt-1.1 {Positional syntax} {
    set res [torch::gt $a $b]
    expr {[string match "tensor*" $res]}
} {1}

test gt-1.2 {Positional syntax result correctness} {
    set res [torch::gt $a $b]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "4"}
} {1}

# -----------------------------------------------------------------------------
# Test 2: Named parameter syntax
# -----------------------------------------------------------------------------

test gt-2.1 {Named syntax with -input1 and -input2} {
    set res [torch::gt -input1 $a -input2 $b]
    expr {[string match "tensor*" $res]}
} {1}

test gt-2.2 {Named syntax with -tensor1 and -tensor2} {
    set res [torch::gt -tensor1 $a -tensor2 $b]
    expr {[string match "tensor*" $res]}
} {1}

test gt-2.3 {Mixed named parameters} {
    set res [torch::gt -input1 $a -tensor2 $b]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 3: camelCase alias
# -----------------------------------------------------------------------------

test gt-3.1 {camelCase alias - positional syntax} {
    set res [torch::Gt $a $b]
    expr {[string match "tensor*" $res]}
} {1}

test gt-3.2 {camelCase alias - named syntax} {
    set res [torch::Gt -input1 $a -input2 $b]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 4: Syntax consistency - both syntaxes produce same results
# -----------------------------------------------------------------------------

test gt-4.1 {Both syntaxes produce same results} {
    set res1 [torch::gt $a $b]
    set res2 [torch::gt -input1 $a -input2 $b]
    set res3 [torch::Gt $a $b]
    set res4 [torch::Gt -input1 $a -input2 $b]
    
    set shape1 [torch::tensor_shape $res1]
    set shape2 [torch::tensor_shape $res2]
    set shape3 [torch::tensor_shape $res3]
    set shape4 [torch::tensor_shape $res4]
    
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4}
} {1}

# -----------------------------------------------------------------------------
# Test 5: Error handling
# -----------------------------------------------------------------------------

test gt-5.1 {Missing arguments} {
    catch {torch::gt} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test gt-5.2 {Only one argument} {
    catch {torch::gt $a} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test gt-5.3 {Invalid tensor handle - first tensor} {
    catch {torch::gt bad_tensor $b} msg
    expr {[string match "*Invalid tensor name*" $msg]}
} {1}

test gt-5.4 {Invalid tensor handle - second tensor} {
    catch {torch::gt $a bad_tensor} msg
    expr {[string match "*Invalid tensor name*" $msg]}
} {1}

test gt-5.5 {Unknown named parameter} {
    catch {torch::gt -foo $a -input2 $b} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test gt-5.6 {Missing value for named parameter} {
    catch {torch::gt -input1 $a -input2} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} {1}

test gt-5.7 {Missing required parameters in named syntax} {
    catch {torch::gt -input1 $a} msg
    expr {[string match "*Required parameters missing*" $msg]}
} {1}

# -----------------------------------------------------------------------------
# Test 6: Basic mathematical correctness
# -----------------------------------------------------------------------------

test gt-6.1 {Mathematical correctness - tensor shape} {
    ;# Test tensors: a = {1 2 3 4}, b = {0 2 4 4}
    ;# Result should be a tensor of same shape
    set res [torch::gt $a $b]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "4"}
} {1}

cleanupTests 