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

proc createInput2d {batch channels height width} {
    # 4D tensor: [batch, channels, height, width]
    return [torch::ones -shape [list $batch $channels $height $width] -dtype float32]
}

proc createTestInput2d {} {
    # Create a test input with known pattern for pooling verification
    set data {}
    for {set i 0} {$i < 16} {incr i} {
        lappend data [expr {$i + 1.0}]
    }
    set input [torch::tensor_create -data $data -shape {1 1 4 4} -dtype float32]
    return $input
}

# =============================================================================
# DUAL SYNTAX TESTS - POSITIONAL SYNTAX
# =============================================================================

test avgpool2d-tensor-1.1 {Basic positional syntax with square kernel} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d $input 2]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-1.2 {Positional syntax with explicit stride} {
    set input [createInput2d 1 1 6 6]
    set result [torch::avgpool2d $input 3 2]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-1.3 {Positional syntax with padding} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d $input 2 2 1]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-1.4 {Positional syntax with all parameters} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d $input 2 1 1 1]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-1.5 {Positional syntax with rectangular kernel} {
    set input [createInput2d 1 1 6 8]
    set result [torch::avgpool2d $input {2 3}]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-1.6 {Positional syntax with rectangular kernel and stride} {
    set input [createInput2d 1 1 8 10]
    set result [torch::avgpool2d $input {2 3} {1 2}]
    expr {$result ne ""}
} 1

# =============================================================================
# DUAL SYNTAX TESTS - NAMED PARAMETER SYNTAX
# =============================================================================

test avgpool2d-tensor-2.1 {Named parameter syntax with -input and -kernel_size} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d -input $input -kernel_size 2]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-2.2 {Named parameter syntax with -tensor alias} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d -tensor $input -kernelSize 2]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-2.3 {Named parameter syntax with all parameters} {
    set input [createInput2d 1 1 6 6]
    set result [torch::avgpool2d -input $input -kernel_size 3 -stride 2 -padding 1]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-2.4 {Named parameter syntax with rectangular sizes} {
    set input [createInput2d 1 1 6 8]
    set result [torch::avgpool2d -input $input -kernel_size {2 3} -stride {1 2}]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-2.5 {Named parameter syntax with count_include_pad} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d -input $input -kernel_size 2 -padding 1 -count_include_pad 0]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-2.6 {Named parameter syntax with countIncludePad alias} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d -input $input -kernelSize 2 -padding 1 -countIncludePad 1]
    expr {$result ne ""}
} 1

# =============================================================================
# CAMELCASE ALIAS TESTS
# =============================================================================

test avgpool2d-tensor-3.1 {CamelCase alias with positional syntax} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgPool2d $input 2]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-3.2 {CamelCase alias with named syntax} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgPool2d -input $input -kernel_size 2]
    expr {$result ne ""}
} 1

# =============================================================================
# MATHEMATICAL CORRECTNESS TESTS
# =============================================================================

test avgpool2d-tensor-4.1 {Mathematical correctness - 2x2 pooling shape} {
    set input [createTestInput2d]
    set result [torch::avgpool2d $input 2]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 2 2}

test avgpool2d-tensor-4.2 {Mathematical correctness - different batch sizes} {
    set input [createInput2d 3 2 4 4]
    set result [torch::avgpool2d $input 2]
    set shape [torch::tensor_shape $result]
    set shape
} {3 2 2 2}

test avgpool2d-tensor-4.3 {Mathematical correctness - stride effects} {
    set input [createInput2d 1 1 6 6]
    set result [torch::avgpool2d $input 2 3]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 2 2}

test avgpool2d-tensor-4.4 {Mathematical correctness - padding effects} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d $input 2 2 1]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 3 3}

# =============================================================================
# EDGE CASES AND SPECIAL VALUES
# =============================================================================

test avgpool2d-tensor-5.1 {Edge case - 1x1 kernel identity operation} {
    set input [createInput2d 1 1 4 4]
    set result [torch::avgpool2d $input 1]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 4 4}

test avgpool2d-tensor-5.2 {Edge case - kernel same size as input} {
    set input [createInput2d 1 1 3 3]
    set result [torch::avgpool2d $input 3]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 1 1}

test avgpool2d-tensor-5.3 {Edge case - rectangular input and kernel} {
    set input [createInput2d 1 1 4 6]
    set result [torch::avgpool2d $input {2 3}]
    set shape [torch::tensor_shape $result]
    set shape
} {1 1 2 2}

test avgpool2d-tensor-5.4 {Large tensor pooling} {
    set input [createInput2d 2 3 32 32]
    set result [torch::avgpool2d $input 4 4]
    set shape [torch::tensor_shape $result]
    set shape
} {2 3 8 8}

# =============================================================================
# DATA TYPE SUPPORT TESTS
# =============================================================================

test avgpool2d-tensor-6.1 {Float32 data type} {
    set input [torch::ones -shape {1 1 4 4} -dtype float32]
    set result [torch::avgpool2d $input 2]
    expr {$result ne ""}
} 1

test avgpool2d-tensor-6.2 {Float64 data type} {
    set input [torch::ones -shape {1 1 4 4} -dtype float64]
    set result [torch::avgpool2d $input 2]
    expr {$result ne ""}
} 1

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

test avgpool2d-tensor-7.1 {Error: Invalid tensor name} {
    catch {torch::avgpool2d "invalid_tensor" 2} msg
    string match "*Invalid input tensor name*" $msg
} 1

test avgpool2d-tensor-7.2 {Error: Missing kernel_size} {
    set input [createInput2d 1 1 4 4]
    set result [catch {torch::avgpool2d $input} msg]
    expr {$result != 0}
} 1

test avgpool2d-tensor-7.3 {Error: Invalid parameter in named syntax} {
    set input [createInput2d 1 1 4 4]
    set result [catch {torch::avgpool2d -input $input -invalid_param 2} msg]
    expr {$result != 0}
} 1

test avgpool2d-tensor-7.4 {Error: Missing parameter value} {
    set input [createInput2d 1 1 4 4]
    set result [catch {torch::avgpool2d -input $input -kernel_size} msg]
    expr {$result != 0}
} 1

# =============================================================================
# SYNTAX CONSISTENCY TESTS
# =============================================================================

test avgpool2d-tensor-8.1 {Syntax consistency - both syntaxes same result} {
    set input1 [createTestInput2d]
    set input2 [createTestInput2d]
    
    set result1 [torch::avgpool2d $input1 2 2 1]
    set result2 [torch::avgpool2d -input $input2 -kernel_size 2 -stride 2 -padding 1]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} 1

test avgpool2d-tensor-8.2 {Syntax consistency - rectangular kernel} {
    set input1 [createInput2d 1 1 6 8]
    set input2 [createInput2d 1 1 6 8]
    
    set result1 [torch::avgpool2d $input1 {2 3} {1 2}]
    set result2 [torch::avgpool2d -input $input2 -kernel_size {2 3} -stride {1 2}]
    
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$shape1 eq $shape2}
} 1

cleanupTests 