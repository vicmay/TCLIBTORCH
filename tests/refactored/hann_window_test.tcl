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

test hann_window-1.1 {Positional syntax - basic window} {
    set res [torch::hann_window 10]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-1.2 {Positional syntax result correctness} {
    set res [torch::hann_window 5]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "5"}
} {1}

# -----------------------------------------------------------------------------
# Test 2: Named parameter syntax
# -----------------------------------------------------------------------------

test hann_window-2.1 {Named syntax with -length} {
    set res [torch::hann_window -length 8]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-2.2 {Named syntax with -window_length} {
    set res [torch::hann_window -window_length 8]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-2.3 {Named syntax with dtype parameter} {
    set res [torch::hann_window -length 8 -dtype float64]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-2.4 {Named syntax with device parameter} {
    set res [torch::hann_window -length 8 -device cpu]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-2.5 {Named syntax with periodic parameter} {
    set res [torch::hann_window -length 8 -periodic true]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-2.6 {Named syntax with all parameters} {
    set res [torch::hann_window -length 8 -dtype float32 -device cpu -periodic false]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 3: camelCase alias
# -----------------------------------------------------------------------------

test hann_window-3.1 {camelCase alias - positional syntax} {
    set res [torch::hannWindow 6]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-3.2 {camelCase alias - named syntax} {
    set res [torch::hannWindow -length 6 -dtype float32]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 4: Syntax consistency - both syntaxes produce same results
# -----------------------------------------------------------------------------

test hann_window-4.1 {Both syntaxes produce same shape} {
    set res1 [torch::hann_window 10]
    set res2 [torch::hann_window -length 10]
    set res3 [torch::hannWindow 10]
    set res4 [torch::hannWindow -length 10]
    
    set shape1 [torch::tensor_shape $res1]
    set shape2 [torch::tensor_shape $res2]
    set shape3 [torch::tensor_shape $res3]
    set shape4 [torch::tensor_shape $res4]
    
    expr {$shape1 eq $shape2 && $shape2 eq $shape3 && $shape3 eq $shape4 && $shape1 eq "10"}
} {1}

# -----------------------------------------------------------------------------
# Test 5: Error handling
# -----------------------------------------------------------------------------

test hann_window-5.1 {Missing arguments} {
    catch {torch::hann_window} msg
    expr {[string match "*Usage*" $msg]}
} {1}

test hann_window-5.2 {Zero window length} {
    catch {torch::hann_window 0} msg
    expr {[string match "*positive*" $msg]}
} {1}

test hann_window-5.3 {Negative window length gets interpreted as parameter} {
    catch {torch::hann_window -5} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} {1}

test hann_window-5.4 {Unknown named parameter} {
    catch {torch::hann_window -foo 8} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test hann_window-5.5 {Missing value for named parameter} {
    catch {torch::hann_window -length} msg
    expr {[string match "*Missing value for parameter*" $msg]}
} {1}

test hann_window-5.6 {Missing required parameters in named syntax} {
    catch {torch::hann_window -dtype float32} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

test hann_window-5.7 {Invalid periodic parameter} {
    catch {torch::hann_window -length 8 -periodic invalid} msg
    expr {[string match "*Invalid periodic parameter*" $msg]}
} {1}

test hann_window-5.8 {Unsupported dtype} {
    catch {torch::hann_window -length 8 -dtype unsupported_type} msg
    expr {[string match "*Unsupported dtype*" $msg]}
} {1}

# -----------------------------------------------------------------------------
# Test 6: Different data types
# -----------------------------------------------------------------------------

test hann_window-6.1 {Float32 dtype} {
    set res [torch::hann_window -length 8 -dtype float32]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-6.2 {Float64 dtype} {
    set res [torch::hann_window -length 8 -dtype float64]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-6.3 {Double dtype alias} {
    set res [torch::hann_window -length 8 -dtype double]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 7: Different window sizes
# -----------------------------------------------------------------------------

test hann_window-7.1 {Small window} {
    set res [torch::hann_window 3]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "3"}
} {1}

test hann_window-7.2 {Medium window} {
    set res [torch::hann_window 64]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "64"}
} {1}

test hann_window-7.3 {Large window} {
    set res [torch::hann_window 1024]
    set shape [torch::tensor_shape $res]
    expr {$shape eq "1024"}
} {1}

# -----------------------------------------------------------------------------
# Test 8: Periodic parameter variations
# -----------------------------------------------------------------------------

test hann_window-8.1 {Periodic true} {
    set res [torch::hann_window -length 8 -periodic true]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-8.2 {Periodic false} {
    set res [torch::hann_window -length 8 -periodic false]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-8.3 {Periodic 1} {
    set res [torch::hann_window -length 8 -periodic 1]
    expr {[string match "tensor*" $res]}
} {1}

test hann_window-8.4 {Periodic 0} {
    set res [torch::hann_window -length 8 -periodic 0]
    expr {[string match "tensor*" $res]}
} {1}

# -----------------------------------------------------------------------------
# Test 9: Comparison with Hamming window (different mathematical properties)
# -----------------------------------------------------------------------------

test hann_window-9.1 {Hann vs Hamming window shapes are same} {
    set hann_win [torch::hann_window 16]
    set hamming_win [torch::hamming_window 16]
    
    set hann_shape [torch::tensor_shape $hann_win]
    set hamming_shape [torch::tensor_shape $hamming_win]
    
    expr {$hann_shape eq $hamming_shape && $hann_shape eq "16"}
} {1}

cleanupTests 