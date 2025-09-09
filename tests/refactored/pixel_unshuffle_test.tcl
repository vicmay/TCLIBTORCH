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
    # Create a 4D tensor (batch_size=1, channels=1, height=4, width=4)
    set data {}
    for {set h 0} {$h < 4} {incr h} {
        for {set w 0} {$w < 4} {incr w} {
            lappend data [expr {$h * 4 + $w + 1}]
        }
    }
    set tensor [torch::tensor_create $data float32 cpu 0]
    return [torch::tensor_reshape $tensor {1 1 4 4}]
}

# Test cases for positional syntax
test pixel_unshuffle-1.1 {Basic positional syntax} {
    set input [create_test_tensor]
    set output [torch::pixel_unshuffle $input 2]
    
    # Verify output shape (1, 4, 2, 2)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 4 2 2}}
} {1}

test pixel_unshuffle-1.2 {Error on missing downscale_factor} {
    set input [create_test_tensor]
    catch {torch::pixel_unshuffle $input} msg
    set msg
} {Usage: torch::pixel_unshuffle input downscale_factor | torch::pixel_unshuffle -input tensor -downscaleFactor int}

test pixel_unshuffle-1.3 {Error on invalid downscale_factor} {
    set input [create_test_tensor]
    catch {torch::pixel_unshuffle $input -1} msg
    set msg
} {Required parameters missing: -input and -downscaleFactor}

# Test cases for named parameter syntax
test pixel_unshuffle-2.1 {Named parameter syntax with -input} {
    set input [create_test_tensor]
    set output [torch::pixel_unshuffle -input $input -downscaleFactor 2]
    
    # Verify output shape (1, 4, 2, 2)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 4 2 2}}
} {1}

test pixel_unshuffle-2.2 {Named parameter syntax with -tensor} {
    set input [create_test_tensor]
    set output [torch::pixel_unshuffle -tensor $input -downscaleFactor 2]
    
    # Verify output shape (1, 4, 2, 2)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 4 2 2}}
} {1}

test pixel_unshuffle-2.3 {Error on missing -input} {
    catch {torch::pixel_unshuffle -downscaleFactor 2} msg
    set msg
} {Required parameters missing: -input and -downscaleFactor}

test pixel_unshuffle-2.4 {Error on invalid -downscaleFactor} {
    set input [create_test_tensor]
    catch {torch::pixel_unshuffle -input $input -downscaleFactor 0} msg
    set msg
} {Required parameters missing: -input and -downscaleFactor}

# Test cases for camelCase alias
test pixel_unshuffle-3.1 {CamelCase alias} {
    set input [create_test_tensor]
    set output [torch::pixelUnshuffle -input $input -downscaleFactor 2]
    
    # Verify output shape (1, 4, 2, 2)
    set shape [torch::tensor_shape $output]
    expr {$shape eq {1 4 2 2}}
} {1}

# Test pixel_shuffle and pixel_unshuffle inverse relationship
test pixel_unshuffle-4.1 {Inverse relationship with pixel_shuffle} {
    # Create original tensor (1, 4, 2, 2)
    set data {}
    for {set c 0} {$c < 4} {incr c} {
        for {set h 0} {$h < 2} {incr h} {
            for {set w 0} {$w < 2} {incr w} {
                lappend data [expr {$c * 4 + $h * 2 + $w + 1}]
            }
        }
    }
    set original [torch::tensor_create $data float32 cpu 0]
    set original [torch::tensor_reshape $original {1 4 2 2}]
    
    # Apply pixel_shuffle (1, 4, 2, 2) -> (1, 1, 4, 4)
    set shuffled [torch::pixel_shuffle $original 2]
    
    # Apply pixel_unshuffle (1, 1, 4, 4) -> (1, 4, 2, 2)
    set unshuffled [torch::pixel_unshuffle $shuffled 2]
    
    # Verify shapes match
    set orig_shape [torch::tensor_shape $original]
    set final_shape [torch::tensor_shape $unshuffled]
    expr {$orig_shape eq $final_shape}
} {1}

cleanupTests 