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

# Test cases for positional syntax (backward compatibility)
test cummax-1.1 {Basic positional syntax - using zeros tensor} {
    set tensor1 [torch::zeros {5} float32 cpu false]
    set result [torch::cummax $tensor1 0]
    
    # Check if result tensor is valid
    expr {[string length $result] > 0}
} {1}

test cummax-1.2 {Positional syntax - 2D tensor} {
    set tensor1 [torch::zeros {3 3} float32 cpu false]
    set result [torch::cummax $tensor1 0]
    
    # Check if result tensor is valid
    expr {[string length $result] > 0}
} {1}

test cummax-1.3 {Positional syntax error - wrong number of arguments} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result [catch {torch::cummax $tensor1} output]
    expr {$result == 1}
} {1}

test cummax-1.4 {Positional syntax error - invalid tensor name} {
    set result [catch {torch::cummax "invalid_tensor" 0} output]
    expr {$result == 1 && [string match "*Invalid tensor name*" $output]}
} {1}

# Test cases for named parameter syntax
test cummax-2.1 {Named parameter syntax - basic usage} {
    set tensor1 [torch::zeros {5} float32 cpu false]
    set result [torch::cummax -input $tensor1 -dim 0]
    
    # Check if result tensor is valid
    expr {[string length $result] > 0}
} {1}

test cummax-2.2 {Named parameter syntax error - missing input} {
    set result [catch {torch::cummax -dim 0} output]
    expr {$result == 1 && [string match "*Required parameter missing: -input*" $output]}
} {1}

test cummax-2.3 {Named parameter syntax error - unknown parameter} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result [catch {torch::cummax -input $tensor1 -unknown_param 0} output]
    expr {$result == 1 && [string match "*Unknown parameter*" $output]}
} {1}

# Test cases for camelCase alias
test cummax-3.1 {CamelCase alias - basic functionality} {
    set tensor1 [torch::zeros {5} float32 cpu false]
    set result [torch::cumMax $tensor1 0]
    
    # Check if result tensor is valid
    expr {[string length $result] > 0}
} {1}

test cummax-3.2 {CamelCase alias with named parameters} {
    set tensor1 [torch::zeros {3 2} float32 cpu false]
    set result [torch::cumMax -input $tensor1 -dim 1]
    
    # Check if result tensor is valid
    expr {[string length $result] > 0}
} {1}

# Test cases for different dimensions
test cummax-4.1 {Different dimensions - dim 0} {
    set tensor1 [torch::zeros {3 4} float32 cpu false]
    set result [torch::cummax $tensor1 0]
    
    # Check if result tensor is valid
    expr {[string length $result] > 0}
} {1}

test cummax-4.2 {Different dimensions - dim 1} {
    set tensor1 [torch::zeros {3 4} float32 cpu false]
    set result [torch::cummax $tensor1 1]
    
    # Check if result tensor is valid
    expr {[string length $result] > 0}
} {1}

test cummax-4.3 {Different dimensions - negative dim} {
    set tensor1 [torch::zeros {3 4} float32 cpu false]
    set result [catch {torch::cummax $tensor1 -1} output]
    
    # Negative dimension should be handled by PyTorch
    expr {$result == 0}
} {1}

# Syntax consistency tests
test cummax-5.1 {Syntax consistency between positional and named} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result1 [torch::cummax $tensor1 0]
    set result2 [torch::cummax -input $tensor1 -dim 0]
    
    # Both should succeed
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test cummax-5.2 {Syntax consistency with camelCase} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result1 [torch::cummax $tensor1 0]
    set result2 [torch::cumMax $tensor1 0]
    
    # Both should succeed
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

test cummax-5.3 {Parameter order independence} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result1 [torch::cummax -input $tensor1 -dim 0]
    set result2 [torch::cummax -dim 0 -input $tensor1]
    
    # Both should succeed
    expr {[string length $result1] > 0 && [string length $result2] > 0}
} {1}

# Error handling tests
test cummax-6.1 {Error handling - missing value for parameter} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result [catch {torch::cummax -input $tensor1 -dim} output]
    expr {$result == 1 && [string match "*Missing value for parameter*" $output]}
} {1}

test cummax-6.2 {Error handling - invalid tensor name in named syntax} {
    set result [catch {torch::cummax -input "invalid_tensor" -dim 0} output]
    expr {$result == 1 && [string match "*Invalid tensor name*" $output]}
} {1}

test cummax-6.3 {Error handling - invalid dimension type} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result [catch {torch::cummax -input $tensor1 -dim "invalid"} output]
    expr {$result == 1}
} {1}

# Integration tests
test cummax-7.1 {Integration - use result in subsequent operations} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set result [torch::cummax $tensor1 0]
    
    # Try to get shape of result (this should work if result is valid)
    set shape [torch::tensor_shape $result]
    expr {[llength $shape] > 0}
} {1}

test cummax-7.2 {Integration - multiple consecutive operations} {
    set tensor1 [torch::zeros {3} float32 cpu false]
    set all_passed 1
    
    for {set i 0} {$i < 3} {incr i} {
        set result [torch::cummax $tensor1 0]
        if {[string length $result] == 0} {
            set all_passed 0
            break
        }
    }
    
    expr {$all_passed}
} {1}

cleanupTests 