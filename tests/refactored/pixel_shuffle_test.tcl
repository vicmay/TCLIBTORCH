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

# Helper function to create a test tensor
proc create_test_tensor {} {
    # Create a 4D tensor (batch_size=1, channels=4, height=2, width=2)
    # Fill with sequential values for easy verification
    set values {}
    for {set c 0} {$c < 4} {incr c} {
        for {set h 0} {$h < 2} {incr h} {
            for {set w 0} {$w < 2} {incr w} {
                lappend values [expr {$c * 4 + $h * 2 + $w + 1}]
            }
        }
    }
    set tensor [torch::tensor_create $values float32]
    return [torch::tensor_reshape $tensor {1 4 2 2}]
}

# Test cases for positional syntax
test pixel_shuffle-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set output [torch::pixel_shuffle $input 2]
    
    # Verify output shape (1, 1, 4, 4)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 1 4 4}}
} {1}

test pixel_shuffle-1.2 {Error on missing upscale_factor} {
    set input [create_test_tensor]
    catch {torch::pixel_shuffle $input} msg
    set msg
} {Usage: torch::pixel_shuffle input upscale_factor | torch::pixel_shuffle -input tensor -upscaleFactor int}

test pixel_shuffle-1.3 {Error on invalid upscale_factor} {
    set input [create_test_tensor]
    catch {torch::pixel_shuffle $input -1} msg
    set msg
} {Required parameters missing: -input and -upscaleFactor}

# Test cases for named parameter syntax
test pixel_shuffle-2.1 {Named parameter syntax with -input} {
    set input [create_test_tensor]
    set output [torch::pixel_shuffle -input $input -upscaleFactor 2]
    
    # Verify output shape (1, 1, 4, 4)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 1 4 4}}
} {1}

test pixel_shuffle-2.2 {Named parameter syntax with -tensor} {
    set input [create_test_tensor]
    set output [torch::pixel_shuffle -tensor $input -upscaleFactor 2]
    
    # Verify output shape (1, 1, 4, 4)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 1 4 4}}
} {1}

test pixel_shuffle-2.3 {Error on missing -input} {
    catch {torch::pixel_shuffle -upscaleFactor 2} msg
    set msg
} {Required parameters missing: -input and -upscaleFactor}

test pixel_shuffle-2.4 {Error on invalid -upscaleFactor} {
    set input [create_test_tensor]
    catch {torch::pixel_shuffle -input $input -upscaleFactor 0} msg
    set msg
} {Required parameters missing: -input and -upscaleFactor}

# Test cases for camelCase alias
test pixel_shuffle-3.1 {CamelCase alias} {
    set input [create_test_tensor]
    set output [torch::pixelShuffle -input $input -upscaleFactor 2]
    
    # Verify output shape (1, 1, 4, 4)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 1 4 4}}
} {1}

cleanupTests
