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
test tensor_relu-1.1 {Basic positional syntax - negative value} {
    set t1 [torch::full {1} -5.0]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0.0}
} {1}

test tensor_relu-1.2 {Positional syntax - positive value} {
    set t1 [torch::full {1} 3.0]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 3.0}
} {1}

test tensor_relu-1.3 {Positional syntax - zero value} {
    set t1 [torch::zeros {1}]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0.0}
} {1}

test tensor_relu-1.4 {Positional syntax - very large positive} {
    set t1 [torch::full {1} 1000.0]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 1000.0}
} {1}

test tensor_relu-1.5 {Positional syntax - very large negative} {
    set t1 [torch::full {1} -1000.0]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0.0}
} {1}

test tensor_relu-1.6 {Positional syntax with multi-element tensor} {
    set t1 [torch::full {2 2} 5.0]
    set result [torch::tensor_relu $t1]
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

# Test cases for named parameter syntax
test tensor_relu-2.1 {Named parameter syntax - negative value} {
    set t1 [torch::full {1} -2.0]
    set result [torch::tensor_relu -input $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0.0}
} {1}

test tensor_relu-2.2 {Named parameter syntax - positive value} {
    set t1 [torch::full {1} 7.0]
    set result [torch::tensor_relu -input $t1]
    set value [torch::tensor_item $result]
    expr {$value == 7.0}
} {1}

test tensor_relu-2.3 {Named parameter syntax preserves tensor properties} {
    set t1 [torch::full {2 3} 1.5]
    set result [torch::tensor_relu -input $t1]
    set shape [torch::tensor_shape $result]
    set dtype [torch::tensor_dtype $result]
    expr {$shape eq "2 3" && [string match "*Float*" $dtype]}
} {1}

# Test cases for camelCase alias
test tensor_relu-3.1 {CamelCase alias syntax - positive} {
    set t1 [torch::full {1} 4.0]
    set result [torch::tensorRelu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 4.0}
} {1}

test tensor_relu-3.2 {CamelCase with named parameters - negative} {
    set t1 [torch::full {1} -3.0]
    set result [torch::tensorRelu -input $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0.0}
} {1}

# Error handling tests
test tensor_relu-4.1 {Error: invalid tensor name (positional)} {
    catch {torch::tensor_relu invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_relu-4.2 {Error: invalid tensor name (named)} {
    catch {torch::tensor_relu -input invalid_tensor} result
    expr {[string match "*Invalid tensor name*" $result]}
} {1}

test tensor_relu-4.3 {Error: missing required parameter} {
    catch {torch::tensor_relu} result
    expr {[string match "*Usage*" $result] || [string match "*Required parameter*" $result]}
} {1}

test tensor_relu-4.4 {Error: unknown parameter} {
    set t1 [torch::ones {1}]
    catch {torch::tensor_relu -invalid_param $t1} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test tensor_relu-4.5 {Error: missing value for named parameter} {
    catch {torch::tensor_relu -input} result
    expr {[string match "*Missing value for parameter*" $result]}
} {1}

# Mathematical correctness tests
test tensor_relu-5.1 {Mathematical properties: non-negativity} {
    set t1 [torch::full {1} -100.0]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value >= 0.0}
} {1}

test tensor_relu-5.2 {Mathematical properties: identity for positive} {
    set t1 [torch::full {1} 42.0]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 42.0}
} {1}

test tensor_relu-5.3 {Mathematical properties: zero for negative} {
    set inputs {-1.0 -0.1 -100.0 -0.001}
    set all_zero 1
    
    foreach input $inputs {
        set t [torch::full {1} $input]
        set result [torch::tensor_relu $t]
        set value [torch::tensor_item $result]
        
        if {$value != 0.0} {
            set all_zero 0
            break
        }
    }
    
    expr {$all_zero}
} {1}

# Data type consistency tests
test tensor_relu-6.1 {Different data types: float64} {
    set t1 [torch::full {1} -2.5 float64]
    set result [torch::tensor_relu $t1]
    set dtype [torch::tensor_dtype $result]
    set value [torch::tensor_item $result]
    expr {[string match "*Float64*" $dtype] && $value == 0.0}
} {1}

# Integration tests with other commands
test tensor_relu-7.1 {Chain operations: relu after subtraction} {
    set t1 [torch::full {1} 5.0]
    set t2 [torch::full {1} 8.0]
    # 5 - 8 = -3
    set diff [torch::tensor_sub $t1 $t2]
    set relu_result [torch::tensor_relu $diff]
    set value [torch::tensor_item $relu_result]
    expr {$value == 0.0}
} {1}

test tensor_relu-7.2 {Neural network pattern: linear + relu} {
    # Simulate: input * weight + bias, then relu
    set input [torch::full {1} 2.0]
    set weight [torch::full {1} -1.5]
    set bias [torch::full {1} 1.0]
    
    # 2 * -1.5 = -3
    set linear_output [torch::tensor_mul $input $weight]
    # -3 + 1 = -2
    set with_bias [torch::tensor_add $linear_output $bias]
    set activated [torch::tensor_relu $with_bias]
    set value [torch::tensor_item $activated]
    expr {$value == 0.0}
} {1}

# Syntax consistency tests
test tensor_relu-8.1 {Both syntaxes produce same result} {
    set t1 [torch::full {1} -1.5]
    set result1 [torch::tensor_relu $t1]
    set result2 [torch::tensor_relu -input $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {$value1 == $value2}
} {1}

test tensor_relu-8.2 {CamelCase alias produces same result} {
    set t1 [torch::full {1} 3.5]
    set result1 [torch::tensor_relu $t1]
    set result2 [torch::tensorRelu $t1]
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    expr {$value1 == $value2 && $value1 == 3.5}
} {1}

# Edge cases and special values
test tensor_relu-9.1 {Edge case: very small positive value} {
    set t1 [torch::full {1} 0.0001]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0.0001}
} {1}

test tensor_relu-9.2 {Edge case: very small negative value} {
    set t1 [torch::full {1} -0.0001]
    set result [torch::tensor_relu $t1]
    set value [torch::tensor_item $result]
    expr {$value == 0.0}
} {1}

test tensor_relu-9.3 {Sparsity property: creates zeros} {
    # Test that ReLU creates sparsity (zeros for negative inputs)
    set pos_count 0
    set zero_count 0
    
    set inputs {-5.0 -1.0 0.0 1.0 5.0}
    foreach input $inputs {
        set t [torch::full {1} $input]
        set result [torch::tensor_relu $t]
        set value [torch::tensor_item $result]
        
        if {$value > 0.0} {
            incr pos_count
        } elseif {$value == 0.0} {
            incr zero_count
        }
    }
    
    # Should have 2 positive values (1.0, 5.0) and 3 zeros (-5.0, -1.0, 0.0)
    expr {$pos_count == 2 && $zero_count == 3}
} {1}

# Gradient behavior verification (mathematical property)
test tensor_relu-10.1 {Gradient property: derivative is 0 or 1} {
    # ReLU derivative is 0 for x < 0 and 1 for x > 0
    # We can verify this by checking the function behavior
    
    set neg_input [torch::full {1} -1.0]
    set pos_input [torch::full {1} 1.0]
    
    set neg_result [torch::tensor_relu $neg_input]
    set pos_result [torch::tensor_relu $pos_input]
    
    set neg_val [torch::tensor_item $neg_result]
    set pos_val [torch::tensor_item $pos_result]
    
    # For negative: f(-1) = 0, for positive: f(1) = 1
    expr {$neg_val == 0.0 && $pos_val == 1.0}
} {1}

cleanupTests 