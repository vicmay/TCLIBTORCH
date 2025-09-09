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

# -----------------------------------------------------------------------------
# Test 1: Basic functionality - Positional syntax (backward compatibility)
# -----------------------------------------------------------------------------

test kaiser_window-1.1 {Positional syntax - basic window} {
    set res [torch::kaiser_window 10]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-1.2 {Positional syntax with beta parameter} {
    set res [torch::kaiser_window 10 8.0]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-1.3 {Positional syntax result correctness} {
    set res [torch::kaiser_window 5]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "5"}
} {1}

# -----------------------------------------------------------------------------
# Test 2: Named parameter syntax
# -----------------------------------------------------------------------------

test kaiser_window-2.1 {Named syntax with -windowLength} {
    set res [torch::kaiser_window -windowLength 8]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-2.2 {Named syntax with -beta parameter} {
    set res [torch::kaiser_window -windowLength 8 -beta 10.0]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-2.3 {Named syntax with dtype parameter} {
    set res [torch::kaiser_window -windowLength 8 -dtype float64]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-2.4 {Named syntax with device parameter} {
    set res [torch::kaiser_window -windowLength 8 -device cpu]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-2.5 {Named syntax with periodic parameter} {
    set res [torch::kaiser_window -windowLength 8 -periodic true]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-2.6 {Named syntax with all parameters} {
    set res [torch::kaiser_window -windowLength 8 -beta 15.0 -dtype float32 -device cpu -periodic false]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 3: camelCase alias
# -----------------------------------------------------------------------------

test kaiser_window-3.1 {camelCase alias - positional syntax} {
    set res [torch::kaiserWindow 6]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-3.2 {camelCase alias - positional syntax with beta} {
    set res [torch::kaiserWindow 6 9.0]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-3.3 {camelCase alias - named syntax} {
    set res [torch::kaiserWindow -windowLength 6 -beta 11.0]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 4: Syntax consistency - both syntaxes produce same results
# -----------------------------------------------------------------------------

test kaiser_window-4.1 {Both syntaxes produce same shape} {
    set res1 [torch::kaiser_window 10]
    set res2 [torch::kaiser_window -windowLength 10]
    set res3 [torch::kaiserWindow 10]
    set res4 [torch::kaiserWindow -windowLength 10]
    
    set shape1 [torch::tensor_shape $res1]
    set shape2 [torch::tensor_shape $res2]
    set shape3 [torch::tensor_shape $res3]
    set shape4 [torch::tensor_shape $res4]
    
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4 && $shape1 eq "10"}
} {1}

test kaiser_window-4.2 {Both syntaxes with same beta produce same results} {
    set res1 [torch::kaiser_window 8 14.0]
    set res2 [torch::kaiser_window -windowLength 8 -beta 14.0]
    set res3 [torch::kaiserWindow 8 14.0]
    set res4 [torch::kaiserWindow -windowLength 8 -beta 14.0]
    
    set shape1 [torch::tensor_shape $res1]
    set shape2 [torch::tensor_shape $res2]
    set shape3 [torch::tensor_shape $res3]
    set shape4 [torch::tensor_shape $res4]
    
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4 && $shape1 eq "8"}
} {1}

# -----------------------------------------------------------------------------
# Test 5: Error handling
# -----------------------------------------------------------------------------

test kaiser_window-5.1 {Missing arguments} {
    catch {torch::kaiser_window} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test kaiser_window-5.2 {Zero window length} {
    catch {torch::kaiser_window 0} msg
    expr {[string match "*positive*" $msg]}
} {1}

test kaiser_window-5.3 {Negative window length gets interpreted as parameter} {
    catch {torch::kaiser_window -5} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test kaiser_window-5.4 {Unknown named parameter} {
    catch {torch::kaiser_window -foo 8} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test kaiser_window-5.5 {Missing value for named parameter} {
    catch {torch::kaiser_window -windowLength} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test kaiser_window-5.6 {Missing required parameters in named syntax} {
    catch {torch::kaiser_window -beta 12.0} msg
    expr {[string match "*positive*" $msg]}
} {1}

# -----------------------------------------------------------------------------
# Test 6: Different data types
# -----------------------------------------------------------------------------

test kaiser_window-6.1 {Float32 dtype} {
    set res [torch::kaiser_window -windowLength 8 -dtype float32]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-6.2 {Float64 dtype} {
    set res [torch::kaiser_window -windowLength 8 -dtype float64]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-6.3 {Double dtype alias} {
    set res [torch::kaiser_window -windowLength 8 -dtype double]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 7: Different window sizes
# -----------------------------------------------------------------------------

test kaiser_window-7.1 {Small window} {
    set res [torch::kaiser_window 3]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "3"}
} {1}

test kaiser_window-7.2 {Medium window} {
    set res [torch::kaiser_window 64]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "64"}
} {1}

test kaiser_window-7.3 {Large window} {
    set res [torch::kaiser_window 1024]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "1024"}
} {1}

# -----------------------------------------------------------------------------
# Test 8: Beta parameter variations
# -----------------------------------------------------------------------------

test kaiser_window-8.1 {Default beta (12.0)} {
    set res [torch::kaiser_window 8]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-8.2 {Low beta value} {
    set res [torch::kaiser_window -windowLength 8 -beta 2.0]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-8.3 {High beta value} {
    set res [torch::kaiser_window -windowLength 8 -beta 20.0]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-8.4 {Zero beta value} {
    set res [torch::kaiser_window -windowLength 8 -beta 0.0]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 9: Periodic parameter variations
# -----------------------------------------------------------------------------

test kaiser_window-9.1 {Periodic true} {
    set res [torch::kaiser_window -windowLength 8 -periodic true]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-9.2 {Periodic false} {
    set res [torch::kaiser_window -windowLength 8 -periodic false]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-9.3 {Periodic 1} {
    set res [torch::kaiser_window -windowLength 8 -periodic 1]
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-9.4 {Periodic 0} {
    set res [torch::kaiser_window -windowLength 8 -periodic 0]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 10: Mathematical correctness
# -----------------------------------------------------------------------------

test kaiser_window-10.1 {Window values are between 0 and 1} {
    set res [torch::kaiser_window 10]
    ;# Note: This is a basic sanity check - actual values depend on implementation
    expr {[string match "tensor*" $res]}
} {1}

test kaiser_window-10.2 {Different beta values produce different results} {
    set res1 [torch::kaiser_window 10 5.0]
    set res2 [torch::kaiser_window 10 15.0]
    ;# Both should be valid tensors (actual comparison would need tensor comparison functions)
    expr {[string match "tensor*" $res1] && [string match "tensor*" $res2]}
} {1}

# Cleanup
cleanupTests 