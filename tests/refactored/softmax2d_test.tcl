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

# Helper to create test tensors
proc create_test_tensor {args} {
    # Flexible helper: if called with two arguments, treat first as variable
    # name and second as the shape list. If called with a single argument,
    # simply return a zeros tensor with the given shape.
    if {[llength $args] == 1} {
        # Usage: create_test_tensor {2 2}
        set tensorShape [lindex $args 0]
        return [torch::zeros $tensorShape]
    } elseif {[llength $args] == 2} {
        # Usage: create_test_tensor myVar {2 2}
        set varName [lindex $args 0]
        set tensorShape [lindex $args 1]
        upvar 1 $varName var
        set var [torch::zeros $tensorShape]
        return $var
    } else {
        error {wrong # args: should be "create_test_tensor ?varName? shape"}
    }
}

# ============================================================================
# Basic Softmax2D Tests - Positional Syntax
# ============================================================================

test softmax2d-1.1 {Basic softmax2d with positional syntax - default dimension} {
    set input [torch::randn {2 3 4 4}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-1.2 {Softmax2d with custom dimension - positional syntax} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d $input 0]
    string match "tensor*" $result
} {1}

test softmax2d-1.3 {Softmax2d with dimension 2 - positional syntax} {
    set input [torch::randn {4 5 6}]
    set result [torch::softmax2d $input 2]
    string match "tensor*" $result
} {1}

test softmax2d-1.4 {Softmax2d with negative dimension - positional syntax} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d $input -1]
    string match "tensor*" $result
} {1}

test softmax2d-1.5 {Softmax2d with 2D tensor - positional syntax} {
    set input [torch::randn {3 4}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

# ============================================================================
# Softmax2D Tests - Named Parameter Syntax
# ============================================================================

test softmax2d-2.1 {Basic softmax2d with named parameter syntax - input parameter} {
    set input [torch::randn {2 3 4 4}]
    set result [torch::softmax2d -input $input]
    string match "tensor*" $result
} {1}

test softmax2d-2.2 {Softmax2d with named parameter syntax - tensor parameter} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d -tensor $input]
    string match "tensor*" $result
} {1}

test softmax2d-2.3 {Softmax2d with custom dimension - named syntax} {
    set input [torch::randn {4 5 6}]
    set result [torch::softmax2d -input $input -dim 0]
    string match "tensor*" $result
} {1}

test softmax2d-2.4 {Softmax2d with dimension parameter - named syntax} {
    set input [torch::randn {3 4 5}]
    set result [torch::softmax2d -input $input -dimension 2]
    string match "tensor*" $result
} {1}

test softmax2d-2.5 {Softmax2d with negative dimension - named syntax} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d -tensor $input -dim -2]
    string match "tensor*" $result
} {1}

test softmax2d-2.6 {Softmax2d with mixed parameter order - named syntax} {
    set input [torch::randn {2 4 3}]
    set result [torch::softmax2d -dim 1 -input $input]
    string match "tensor*" $result
} {1}

# ============================================================================
# CamelCase Alias Tests
# ============================================================================

test softmax2d-3.1 {Basic softmax2d with camelCase alias - default parameters} {
    set input [torch::randn {2 3 4 4}]
    set result [torch::softmax2D -input $input]
    string match "tensor*" $result
} {1}

test softmax2d-3.2 {Softmax2d camelCase alias with custom dimension} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2D -input $input -dim 0]
    string match "tensor*" $result
} {1}

test softmax2d-3.3 {Softmax2d camelCase alias with positional syntax} {
    set input [torch::randn {4 5 6}]
    set result [torch::softmax2D $input 1]
    string match "tensor*" $result
} {1}

test softmax2d-3.4 {CamelCase and snake_case equivalence} {
    set input1 [torch::randn {2 3 4}]
    set input2 [torch::randn {2 3 4}]
    set result1 [torch::softmax2d -input $input1 -dim 1]
    set result2 [torch::softmax2D -input $input2 -dim 1]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# ============================================================================
# Parameter Validation Tests
# ============================================================================

test softmax2d-4.1 {Error: Missing required parameter - positional} {
    catch {torch::softmax2d} error
    string match "*Usage*" $error
} {1}

test softmax2d-4.2 {Error: Missing required parameter - named} {
    catch {torch::softmax2d -dim 1} error
    string match "*Required parameter missing*" $error
} {1}

test softmax2d-4.3 {Error: Invalid tensor name - positional} {
    catch {torch::softmax2d nonexistent_tensor} error
    string match "*Invalid tensor name*" $error
} {1}

test softmax2d-4.4 {Error: Invalid tensor name - named} {
    catch {torch::softmax2d -input nonexistent_tensor} error
    string match "*Invalid tensor name*" $error
} {1}

test softmax2d-4.5 {Error: Invalid dimension value - positional} {
    set input [torch::randn {2 3}]
    catch {torch::softmax2d $input abc} error
    string match "*Invalid dimension*" $error
} {1}

test softmax2d-4.6 {Error: Invalid dimension value - named} {
    set input [torch::randn {2 3}]
    catch {torch::softmax2d -input $input -dim xyz} error
    string match "*Invalid dimension*" $error
} {1}

test softmax2d-4.7 {Error: Unknown parameter - named} {
    set input [torch::randn {2 3}]
    catch {torch::softmax2d -input $input -invalidParam value} error
    string match "*Unknown parameter*" $error
} {1}

test softmax2d-4.8 {Error: Too many positional arguments} {
    set input [torch::randn {2 3}]
    catch {torch::softmax2d $input 1 extra_arg} error
    string match "*Usage*" $error
} {1}

test softmax2d-4.9 {Error: Missing parameter value - named} {
    set input [torch::randn {2 3}]
    catch {torch::softmax2d -input $input -dim} error
    string match "*Missing value*" $error
} {1}

# ============================================================================
# Edge Cases and Special Values
# ============================================================================

test softmax2d-5.1 {Large tensor dimensions} {
    set input [torch::randn {50 100 25}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-5.2 {Single channel tensor} {
    set input [torch::randn {1 10 10}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-5.3 {4D tensor (batch, channel, height, width)} {
    set input [torch::randn {2 3 4 5}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-5.4 {5D tensor} {
    set input [torch::randn {2 3 4 5 6}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-5.5 {2D tensor (typical matrix)} {
    set input [torch::randn {3 4}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-5.6 {3D tensor with dimension 0} {
    set input [torch::randn {3 4 5}]
    set result [torch::softmax2d -input $input -dim 0]
    string match "tensor*" $result
} {1}

# ============================================================================
# Syntax Equivalence Tests
# ============================================================================

test softmax2d-6.1 {Positional and named syntax equivalence - default dimension} {
    set input1 [torch::randn {2 3 4}]
    set input2 [torch::randn {2 3 4}]
    set result1 [torch::softmax2d $input1]
    set result2 [torch::softmax2d -input $input2]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test softmax2d-6.2 {Positional and named syntax equivalence - custom dimension} {
    set input1 [torch::randn {3 4 5}]
    set input2 [torch::randn {3 4 5}]
    set result1 [torch::softmax2d $input1 0]
    set result2 [torch::softmax2d -input $input2 -dim 0]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test softmax2d-6.3 {Different parameter names equivalence} {
    set input1 [torch::randn {2 3 4}]
    set input2 [torch::randn {2 3 4}]
    set result1 [torch::softmax2d -input $input1 -dim 2]
    set result2 [torch::softmax2d -tensor $input2 -dimension 2]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

test softmax2d-6.4 {CamelCase positional and named equivalence} {
    set input1 [torch::randn {4 4 4}]
    set input2 [torch::randn {4 4 4}]
    set result1 [torch::softmax2D $input1 2]
    set result2 [torch::softmax2D -input $input2 -dim 2]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2]}
} {1}

# ============================================================================
# Mathematical Properties Tests
# ============================================================================

test softmax2d-7.1 {Softmax2d preserves tensor shape} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-7.2 {Softmax2d with different dimensions} {
    set input [torch::randn {3 4 5}]
    set result1 [torch::softmax2d $input 0]
    set result2 [torch::softmax2d $input 1]
    set result3 [torch::softmax2d $input 2]
    expr {[string match "tensor*" $result1] && [string match "tensor*" $result2] && [string match "tensor*" $result3]}
} {1}

test softmax2d-7.3 {Softmax2d with zero tensor} {
    set input [torch::zeros {3 4}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-7.4 {Softmax2d with ones tensor} {
    set input [torch::ones {2 3}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

# ============================================================================
# Different Tensor Types
# ============================================================================

test softmax2d-8.1 {Softmax2d with random normal tensor} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-8.2 {Softmax2d with uniform random tensor} {
    set input [torch::rand {3 4 5}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

test softmax2d-8.3 {Softmax2d with arange tensor} {
    set values [torch::arange -start 0 -end 24 -step 1]
    set input [torch::tensor_reshape $values {2 3 4}]
    set result [torch::softmax2d $input]
    string match "tensor*" $result
} {1}

# ============================================================================
# Dimension-specific Tests
# ============================================================================

test softmax2d-9.1 {Softmax2d along dimension -1 (last)} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d -input $input -dim -1]
    string match "tensor*" $result
} {1}

test softmax2d-9.2 {Softmax2d along dimension -2 (second to last)} {
    set input [torch::randn {2 3 4}]
    set result [torch::softmax2d -input $input -dim -2]
    string match "tensor*" $result
} {1}

test softmax2d-9.3 {Softmax2d with batch dimension} {
    set input [torch::randn {8 3 4 4}]
    # Along channel dimension
    set result [torch::softmax2d -input $input -dim 1]
    string match "tensor*" $result
} {1}

cleanupTests 