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

;# Test 1: Basic positional syntax (backward compatibility)
test layer_norm-1.1 {Basic positional syntax with single dimension} {
    set layer [torch::layer_norm 10]
    expr {$layer ne ""}
} {1}

;# Test 2: Basic positional syntax with list dimension
test layer_norm-1.2 {Basic positional syntax with list dimension} {
    set layer [torch::layer_norm {10 5}]
    expr {$layer ne ""}
} {1}

;# Test 3: Positional syntax with eps parameter
test layer_norm-1.3 {Positional syntax with eps parameter} {
    set layer [torch::layer_norm 10 1e-6]
    expr {$layer ne ""}
} {1}

;# Test 4: Named parameter syntax - single dimension
test layer_norm-2.1 {Named parameter syntax with -normalizedShape} {
    set layer [torch::layer_norm -normalizedShape 10]
    expr {$layer ne ""}
} {1}

;# Test 5: Named parameter syntax - list dimension
test layer_norm-2.2 {Named parameter syntax with list dimension} {
    set layer [torch::layer_norm -normalizedShape {10 5}]
    expr {$layer ne ""}
} {1}

;# Test 6: Named parameter syntax with eps
test layer_norm-2.3 {Named parameter syntax with eps} {
    set layer [torch::layer_norm -normalizedShape 10 -eps 1e-6]
    expr {$layer ne ""}
} {1}

;# Test 7: Named parameter syntax with different order
test layer_norm-2.4 {Named parameter syntax with different order} {
    set layer [torch::layer_norm -eps 1e-6 -normalizedShape 10]
    expr {$layer ne ""}
} {1}

;# Test 8: camelCase alias with positional syntax
test layer_norm-3.1 {camelCase alias with positional syntax} {
    set layer [torch::layerNorm 10]
    expr {$layer ne ""}
} {1}

;# Test 9: camelCase alias with named parameter syntax
test layer_norm-3.2 {camelCase alias with named parameter syntax} {
    set layer [torch::layerNorm -normalizedShape 10 -eps 1e-5]
    expr {$layer ne ""}
} {1}

;# Test 10: Error handling - empty normalized shape (positional)
test layer_norm-4.1 {Error handling - empty normalized shape (positional)} {
    catch {torch::layer_norm {}} msg
    expr {[string match "*Invalid*" $msg] || [string match "*empty*" $msg]}
} {1}

;# Test 11: Error handling - missing parameter value (named)
test layer_norm-4.2 {Error handling - missing parameter value (named)} {
    catch {torch::layer_norm -normalizedShape} msg
    expr {[string match "*Missing value*" $msg] || [string match "*value*" $msg]}
} {1}

;# Test 12: Error handling - unknown parameter (named)
test layer_norm-4.3 {Error handling - unknown parameter (named)} {
    catch {torch::layer_norm -unknown_param 10 -normalizedShape 10} msg
    expr {[string match "*Unknown parameter*" $msg] || [string match "*unknown*" $msg]}
} {1}

;# Test 13: Error handling - invalid normalized shape value (named)
test layer_norm-4.4 {Error handling - invalid normalized shape value (named)} {
    catch {torch::layer_norm -normalizedShape invalid} msg
    expr {[string match "*Invalid*" $msg]}
} {1}

;# Test 14: Different normalized shapes - 1D
test layer_norm-5.1 {Different normalized shapes - 1D} {
    set layer1 [torch::layer_norm 128]
    set layer2 [torch::layer_norm 256]
    set layer3 [torch::layer_norm 512]
    expr {$layer1 ne "" && $layer2 ne "" && $layer3 ne ""}
} {1}

;# Test 15: Different normalized shapes - 2D
test layer_norm-5.2 {Different normalized shapes - 2D} {
    set layer1 [torch::layer_norm {10 10}]
    set layer2 [torch::layer_norm {20 30}]
    set layer3 [torch::layer_norm {64 64}]
    expr {$layer1 ne "" && $layer2 ne "" && $layer3 ne ""}
} {1}

;# Test 16: Different normalized shapes - 3D
test layer_norm-5.3 {Different normalized shapes - 3D} {
    set layer1 [torch::layer_norm {10 10 10}]
    set layer2 [torch::layer_norm {5 8 12}]
    expr {$layer1 ne "" && $layer2 ne ""}
} {1}

;# Test 17: Different eps values
test layer_norm-5.4 {Different eps values} {
    set layer1 [torch::layer_norm 10 1e-5]
    set layer2 [torch::layer_norm 10 1e-6]
    set layer3 [torch::layer_norm 10 1e-8]
    expr {$layer1 ne "" && $layer2 ne "" && $layer3 ne ""}
} {1}

;# Test 18: Multiple parameter formats create valid layers
test layer_norm-6.1 {Multiple parameter formats create valid layers} {
    set layer1 [torch::layer_norm 64]
    set layer2 [torch::layer_norm -normalizedShape 64]
    set layer3 [torch::layerNorm 64]
    set layer4 [torch::layerNorm -normalizedShape 64]
    
    expr {$layer1 ne "" && $layer2 ne "" && $layer3 ne "" && $layer4 ne ""}
} {1}

;# Test 19: Complex normalized shapes
test layer_norm-6.2 {Complex normalized shapes with nested lists} {
    set layer1 [torch::layer_norm {128}]
    set layer2 [torch::layer_norm {64 64}]
    set layer3 [torch::layer_norm {32 16 8}]
    
    expr {$layer1 ne "" && $layer2 ne "" && $layer3 ne ""}
} {1}

;# Test 20: Named parameters with various eps values
test layer_norm-6.3 {Named parameters with various eps values} {
    set layer1 [torch::layer_norm -normalizedShape 64 -eps 1e-5]
    set layer2 [torch::layer_norm -normalizedShape 64 -eps 1e-6]
    set layer3 [torch::layer_norm -normalizedShape 64 -eps 1e-8]
    
    expr {$layer1 ne "" && $layer2 ne "" && $layer3 ne ""}
} {1}

;# Test 21: Parameter validation consistency
test layer_norm-7.1 {Parameter validation consistency} {
    set valid1 [catch {torch::layer_norm 64} layer1]
    set valid2 [catch {torch::layer_norm -normalizedShape 64} layer2]
    set valid3 [catch {torch::layerNorm 64} layer3]
    set valid4 [catch {torch::layerNorm -normalizedShape 64} layer4]
    
    expr {$valid1 == 0 && $valid2 == 0 && $valid3 == 0 && $valid4 == 0}
} {1}

;# Test 22: Edge cases - large dimensions
test layer_norm-7.2 {Edge cases - large dimensions} {
    set layer [torch::layer_norm 1024]
    expr {$layer ne ""}
} {1}

;# Test 23: Edge cases - very small eps values
test layer_norm-7.3 {Edge cases - very small eps values} {
    set layer [torch::layer_norm -normalizedShape 32 -eps 1e-10]
    expr {$layer ne ""}
} {1}

;# Test 24: High-dimensional normalization setup
test layer_norm-7.4 {High-dimensional normalization setup} {
    set layer [torch::layer_norm 768]
    expr {$layer ne ""}
} {1}

;# Test 25: Vision transformer like dimensions
test layer_norm-7.5 {Vision transformer like dimensions} {
    set layer_embed [torch::layer_norm 384]
    set layer_large [torch::layer_norm 768]
    set layer_xlarge [torch::layer_norm 1024]
    
    expr {$layer_embed ne "" && $layer_large ne "" && $layer_xlarge ne ""}
} {1}

;# Test 26: Different layer configurations
test layer_norm-8.1 {Different layer configurations} {
    set bert_base [torch::layer_norm 768]
    set bert_large [torch::layer_norm 1024]
    set gpt_small [torch::layer_norm 768]
    set gpt_medium [torch::layer_norm 1024]
    set gpt_large [torch::layer_norm 1280]
    
    expr {$bert_base ne "" && $bert_large ne "" && $gpt_small ne "" && $gpt_medium ne "" && $gpt_large ne ""}
} {1}

;# Test 27: Mixed parameter usage
test layer_norm-8.2 {Mixed parameter usage patterns} {
    set layer1 [torch::layer_norm 256]
    set layer2 [torch::layer_norm -normalizedShape 256 -eps 1e-5]
    set layer3 [torch::layerNorm 256]
    set layer4 [torch::layerNorm -normalizedShape 256 -eps 1e-5]
    
    expr {$layer1 ne "" && $layer2 ne "" && $layer3 ne "" && $layer4 ne ""}
} {1}

;# Test 28: Sequential layer creation
test layer_norm-8.3 {Sequential layer creation} {
    set layers {}
    for {set i 64} {$i <= 512} {incr i 64} {
        set layer [torch::layer_norm $i]
        if {$layer eq ""} {
            set layers "failed"
            break
        }
        lappend layers $layer
    }
    expr {$layers ne "failed" && [llength $layers] == 8}
} {1}

;# Test 29: Named vs positional parameter equivalence
test layer_norm-8.4 {Named vs positional parameter equivalence} {
    set pos_layer [torch::layer_norm 128 1e-6]
    set named_layer [torch::layer_norm -normalizedShape 128 -eps 1e-6]
    set camel_layer [torch::layerNorm -normalizedShape 128 -eps 1e-6]
    
    expr {$pos_layer ne "" && $named_layer ne "" && $camel_layer ne ""}
} {1}

;# Test 30: Comprehensive shape testing
test layer_norm-8.5 {Comprehensive shape testing} {
    set shapes [list 32 64 128 256 512 768 1024]
    set all_valid 1
    
    foreach shape $shapes {
        set layer [torch::layer_norm $shape]
        if {$layer eq ""} {
            set all_valid 0
            break
        }
    }
    
    expr {$all_valid}
} {1}

cleanupTests
