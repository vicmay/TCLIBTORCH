#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# ============================================================================
;# TORCH::GRID_SAMPLE TESTS
;# Test both positional and named parameter syntax
;# ============================================================================

;# Helper function to create test input and grid tensors
proc create_test_tensors {} {
    ;# Create a simple 4D input tensor (batch=1, channels=1, height=4, width=4)
    set input_data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0}
    set input [torch::tensor_create -data $input_data -shape {1 1 4 4} -dtype float32]
    
    ;# Create a simple 4D grid tensor (batch=1, height=2, width=2, coordinates=2)
    ;# Grid values should be in range [-1, 1] for normalized coordinates
    set grid_data {-0.5 -0.5  0.5 -0.5  -0.5  0.5   0.5  0.5}
    set grid [torch::tensor_create -data $grid_data -shape {1 2 2 2} -dtype float32]
    
    return [list $input $grid]
}

;# Test torch::grid_sample - Positional Syntax (Backward Compatibility)
test grid_sample-1.1 {Basic positional syntax} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample $input $grid]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-1.2 {Positional syntax with mode parameter} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample $input $grid bilinear]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-1.3 {Positional syntax with mode and padding_mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample $input $grid nearest zeros]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-1.4 {Positional syntax with all parameters} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample $input $grid bilinear border 1]
    torch::tensor_shape $result
} -result {1 1 2 2}

;# Test torch::grid_sample - Named Parameter Syntax
test grid_sample-2.1 {Named parameter syntax with -input and -grid} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-2.2 {Named parameter syntax with -tensor} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -tensor $input -grid $grid]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-2.3 {Named parameter syntax with -mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -mode nearest]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-2.4 {Named parameter syntax with -padding_mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -padding_mode border]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-2.5 {Named parameter syntax with -paddingMode (camelCase)} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -paddingMode reflection]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-2.6 {Named parameter syntax with -align_corners} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -align_corners 1]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-2.7 {Named parameter syntax with -alignCorners (camelCase)} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -alignCorners 0]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-2.8 {Named parameter syntax with all parameters} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -mode bilinear -padding_mode zeros -align_corners 1]
    torch::tensor_shape $result
} -result {1 1 2 2}

;# Test torch::gridSample - camelCase alias
test grid_sample-3.1 {camelCase alias with positional syntax} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::gridSample $input $grid]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-3.2 {camelCase alias with named syntax} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::gridSample -input $input -grid $grid]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-3.3 {camelCase alias with mode parameter} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::gridSample -input $input -grid $grid -mode nearest]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-3.4 {camelCase alias with all parameters} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::gridSample -input $input -grid $grid -mode bilinear -paddingMode border -alignCorners 1]
    torch::tensor_shape $result
} -result {1 1 2 2}

;# Test different interpolation modes
test grid_sample-4.1 {Bilinear interpolation mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -mode bilinear]
    torch::tensor_numel $result
} -result 4

test grid_sample-4.2 {Nearest interpolation mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -mode nearest]
    torch::tensor_numel $result
} -result 4

;# Test different padding modes
test grid_sample-5.1 {Zeros padding mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -padding_mode zeros]
    torch::tensor_numel $result
} -result 4

test grid_sample-5.2 {Border padding mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -padding_mode border]
    torch::tensor_numel $result
} -result 4

test grid_sample-5.3 {Reflection padding mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result [torch::grid_sample -input $input -grid $grid -padding_mode reflection]
    torch::tensor_numel $result
} -result 4

;# Test syntax consistency - both syntaxes should produce same results
test grid_sample-6.1 {Syntax consistency - basic grid sample} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result1 [torch::grid_sample $input $grid]
    set result2 [torch::grid_sample -input $input -grid $grid]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

test grid_sample-6.2 {Syntax consistency - with mode parameter} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result1 [torch::grid_sample $input $grid nearest]
    set result2 [torch::grid_sample -input $input -grid $grid -mode nearest]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

test grid_sample-6.3 {Syntax consistency - camelCase alias} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result1 [torch::grid_sample $input $grid]
    set result2 [torch::gridSample $input $grid]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

test grid_sample-6.4 {Syntax consistency - all parameters} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    set result1 [torch::grid_sample $input $grid bilinear border 1]
    set result2 [torch::grid_sample -input $input -grid $grid -mode bilinear -padding_mode border -align_corners 1]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} -result 1

;# Test error handling
test grid_sample-7.1 {Error handling - missing input} -body {
    catch {torch::grid_sample} result
    expr {[string match "*Required parameters missing*" $result]}
} -result 1

test grid_sample-7.2 {Error handling - missing grid} -body {
    set input [torch::ones {1 1 4 4}]
    catch {torch::grid_sample -input $input} result
    expr {[string match "*Required parameters missing*" $result]}
} -result 1

test grid_sample-7.3 {Error handling - invalid tensor name} -body {
    set grid [torch::ones {1 2 2 2}]
    catch {torch::grid_sample "invalid_tensor" $grid} result
    expr {[string match "*Invalid input tensor*" $result]}
} -result 1

test grid_sample-7.4 {Error handling - invalid grid tensor name} -body {
    set input [torch::ones {1 1 4 4}]
    catch {torch::grid_sample $input "invalid_grid"} result
    expr {[string match "*Invalid grid tensor*" $result]}
} -result 1

test grid_sample-7.5 {Error handling - invalid mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    catch {torch::grid_sample -input $input -grid $grid -mode invalid_mode} result
    expr {[string match "*Invalid mode*" $result]}
} -result 1

test grid_sample-7.6 {Error handling - invalid padding mode} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    catch {torch::grid_sample -input $input -grid $grid -padding_mode invalid_padding} result
    expr {[string match "*Invalid padding_mode*" $result]}
} -result 1

test grid_sample-7.7 {Error handling - invalid align_corners parameter} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    catch {torch::grid_sample -input $input -grid $grid -align_corners "invalid"} result
    expr {[string match "*Invalid align_corners parameter*" $result]}
} -result 1

test grid_sample-7.8 {Error handling - unknown parameter} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    catch {torch::grid_sample -input $input -grid $grid -unknown_param value} result
    expr {[string match "*Unknown parameter*" $result]}
} -result 1

test grid_sample-7.9 {Error handling - missing parameter value} -body {
    set tensors [create_test_tensors]
    set input [lindex $tensors 0]
    set grid [lindex $tensors 1]
    catch {torch::grid_sample -input $input -grid} result
    expr {[string match "*Named parameters must come in pairs*" $result]}
} -result 1

;# Test different data types
test grid_sample-8.1 {Different data types - float64} -body {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float64]
    set grid [torch::tensor_create -data {0.0 0.0} -shape {1 1 1 2} -dtype float64]
    set result [torch::grid_sample $input $grid]
    torch::tensor_shape $result
} -result {1 1 1 1}

test grid_sample-8.2 {Different data types - float32 input with float32 grid} -body {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]
    set grid [torch::tensor_create -data {0.0 0.0} -shape {1 1 1 2} -dtype float32]
    set result [torch::grid_sample $input $grid]
    torch::tensor_shape $result
} -result {1 1 1 1}

;# Test edge cases
test grid_sample-9.1 {Edge case - minimal grid sample} -body {
    set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]
    set grid [torch::tensor_create -data {0.0 0.0} -shape {1 1 1 2} -dtype float32]
    set result [torch::grid_sample $input $grid]
    torch::tensor_numel $result
} -result 1

test grid_sample-9.2 {Edge case - larger input tensor} -body {
    set data [list]
    for {set i 0} {$i < 64} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set input [torch::tensor_create -data $data -shape {1 1 8 8} -dtype float32]
    set grid [torch::tensor_create -data {-0.5 -0.5  0.5 -0.5  -0.5  0.5   0.5  0.5} -shape {1 2 2 2} -dtype float32]
    set result [torch::grid_sample $input $grid]
    torch::tensor_shape $result
} -result {1 1 2 2}

test grid_sample-9.3 {Edge case - multi-channel input} -body {
    set data [list]
    for {set i 0} {$i < 32} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set input [torch::tensor_create -data $data -shape {1 2 4 4} -dtype float32]
    set grid [torch::tensor_create -data {-0.5 -0.5  0.5 -0.5  -0.5  0.5   0.5  0.5} -shape {1 2 2 2} -dtype float32]
    set result [torch::grid_sample $input $grid]
    torch::tensor_shape $result
} -result {1 2 2 2}

test grid_sample-9.4 {Edge case - batch processing} -body {
    set data [list]
    for {set i 0} {$i < 32} {incr i} {
        lappend data [expr {$i * 1.0}]
    }
    set input [torch::tensor_create -data $data -shape {2 1 4 4} -dtype float32]
    set grid_data [list]
    for {set i 0} {$i < 16} {incr i} {
        lappend grid_data [expr {($i % 2) * 0.5 - 0.25}]
    }
    set grid [torch::tensor_create -data $grid_data -shape {2 2 2 2} -dtype float32]
    set result [torch::grid_sample $input $grid]
    torch::tensor_shape $result
} -result {2 1 2 2}

cleanupTests 