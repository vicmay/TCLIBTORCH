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

# ============================================================================
# Test torch::grad_scaler_scale - Positional Syntax (Backward Compatibility)
# ============================================================================

test grad_scaler_scale-1.1 {Basic positional syntax} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    set scaled [torch::grad_scaler_scale $scaler $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-1.2 {Positional syntax with custom scaler} {
    set scaler [torch::grad_scaler_new 1024.0]
    set tensor [torch::tensor_create {0.5 1.5 2.5}]
    set scaled [torch::grad_scaler_scale $scaler $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-1.3 {Positional syntax with different tensor} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0}]
    set scaled [torch::grad_scaler_scale $scaler $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-1.4 {Positional syntax with multiple operations} {
    set scaler [torch::grad_scaler_new 512.0]
    set tensor1 [torch::tensor_create {1.0 2.0}]
    set tensor2 [torch::tensor_create {3.0 4.0}]
    set scaled1 [torch::grad_scaler_scale $scaler $tensor1]
    set scaled2 [torch::grad_scaler_scale $scaler $tensor2]
    set result [expr {[string match "tensor*" $scaled1] && [string match "tensor*" $scaled2]}]
    unset scaler tensor1 tensor2 scaled1 scaled2
    set result
} {1}

# ============================================================================
# Test torch::grad_scaler_scale - Named Parameter Syntax
# ============================================================================

test grad_scaler_scale-2.1 {Named parameter syntax - basic usage} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    set scaled [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-2.2 {Named parameter syntax - parameters in different order} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {0.5 1.5 2.5}]
    set scaled [torch::grad_scaler_scale -tensor $tensor -scaler $scaler]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-2.3 {Named parameter syntax - with custom scaler settings} {
    set scaler [torch::grad_scaler_new -initScale 2048.0]
    set tensor [torch::tensor_create {1.0 2.0}]
    set scaled [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-2.4 {Named parameter syntax - gradScaler alias} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    set scaled [torch::grad_scaler_scale -gradScaler $scaler -tensor $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-2.5 {Named parameter syntax - input alias} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    set scaled [torch::grad_scaler_scale -scaler $scaler -input $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

# ============================================================================
# Test torch::gradScalerScale - CamelCase Alias
# ============================================================================

test grad_scaler_scale-3.1 {CamelCase alias - positional syntax} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    set scaled [torch::gradScalerScale $scaler $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-3.2 {CamelCase alias - named parameter syntax} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    set scaled [torch::gradScalerScale -scaler $scaler -tensor $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-3.3 {CamelCase alias - with custom parameters} {
    set scaler [torch::gradScalerNew -initScale 4096.0]
    set tensor [torch::tensor_create {0.1 0.2 0.3}]
    set scaled [torch::gradScalerScale -scaler $scaler -tensor $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

# ============================================================================
# Test Error Handling
# ============================================================================

test grad_scaler_scale-4.1 {Error handling - invalid scaler handle} {
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::grad_scaler_scale -scaler invalid_scaler -tensor $tensor} result
    expr {[string match "*not found*" $result]}
} {1}

test grad_scaler_scale-4.2 {Error handling - invalid tensor handle} {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_scale -scaler $scaler -tensor invalid_tensor} result
    expr {[string match "*not found*" $result]}
} {1}

test grad_scaler_scale-4.3 {Error handling - missing scaler parameter} {
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::grad_scaler_scale -tensor $tensor} result
    expr {[string match "*missing*" $result]}
} {1}

test grad_scaler_scale-4.4 {Error handling - missing tensor parameter} {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_scale -scaler $scaler} result
    expr {[string match "*missing*" $result]}
} {1}

test grad_scaler_scale-4.5 {Error handling - unknown parameter} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::grad_scaler_scale -invalidParam value -scaler $scaler -tensor $tensor} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test grad_scaler_scale-4.6 {Error handling - odd number of parameters} {
    catch {torch::grad_scaler_scale -scaler} result
    expr {[string match "*pairs*" $result]}
} {1}

test grad_scaler_scale-4.7 {Error handling - too few positional parameters} {
    set scaler [torch::grad_scaler_new]
    catch {torch::grad_scaler_scale $scaler} result
    expr {[string match "*Usage*" $result]}
} {1}

test grad_scaler_scale-4.8 {Error handling - too many positional parameters} {
    set scaler [torch::grad_scaler_new]
    set tensor [torch::tensor_create {1.0 2.0 3.0}]
    catch {torch::grad_scaler_scale $scaler $tensor extra} result
    expr {[string match "*Usage*" $result]}
} {1}

# ============================================================================
# Test Integration with Other Functions
# ============================================================================

test grad_scaler_scale-5.1 {Integration - use with tensor operations} {
    set scaler [torch::grad_scaler_new -initScale 2.0]
    set tensor1 [torch::tensor_create {1.0 2.0}]
    set tensor2 [torch::tensor_create {3.0 4.0}]
    set scaled1 [torch::grad_scaler_scale -scaler $scaler -tensor $tensor1]
    set scaled2 [torch::grad_scaler_scale -scaler $scaler -tensor $tensor2]
    set result [torch::tensor_add $scaled1 $scaled2]
    set result_name [string match "tensor*" $result]
    unset scaler tensor1 tensor2 scaled1 scaled2 result
    set result_name
} {1}

test grad_scaler_scale-5.2 {Integration - verify scale value from scaler} {
    set scaler [torch::grad_scaler_new -initScale 8.0]
    set scale_value [torch::grad_scaler_get_scale $scaler]
    set tensor [torch::tensor_create {1.0}]
    set scaled [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    # Verify scale value is what we expect
    expr {$scale_value == 8.0}
} {1}

test grad_scaler_scale-5.3 {Integration - both syntaxes create tensors} {
    set scaler [torch::grad_scaler_new -initScale 3.0]
    set tensor [torch::tensor_create {2.0 4.0}]
    set scaled_pos [torch::grad_scaler_scale $scaler $tensor]
    set scaled_named [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    # Both should create valid tensor handles
    expr {[string match "tensor*" $scaled_pos] && [string match "tensor*" $scaled_named]}
} {1}

# ============================================================================
# Test Edge Cases
# ============================================================================

test grad_scaler_scale-6.1 {Edge case - very small scale factor} {
    set scaler [torch::grad_scaler_new -initScale 0.001]
    set tensor [torch::tensor_create {1000.0}]
    set scaled [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-6.2 {Edge case - very large scale factor} {
    set scaler [torch::grad_scaler_new -initScale 1000000.0]
    set tensor [torch::tensor_create {0.000001}]
    set scaled [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]
    set result [string match "tensor*" $scaled]
    unset scaler tensor scaled
    set result
} {1}

test grad_scaler_scale-6.3 {Edge case - multiple scalers with different tensors} {
    set scaler1 [torch::grad_scaler_new -initScale 1.0]
    set scaler2 [torch::grad_scaler_new -initScale 2.0]
    set scaler3 [torch::grad_scaler_new -initScale 3.0]
    set tensor1 [torch::tensor_create {1.0}]
    set tensor2 [torch::tensor_create {2.0}] 
    set tensor3 [torch::tensor_create {3.0}]
    set scaled1 [torch::grad_scaler_scale -scaler $scaler1 -tensor $tensor1]
    set scaled2 [torch::grad_scaler_scale -scaler $scaler2 -tensor $tensor2]
    set scaled3 [torch::grad_scaler_scale -scaler $scaler3 -tensor $tensor3]
    set result [expr {[string match "tensor*" $scaled1] && [string match "tensor*" $scaled2] && [string match "tensor*" $scaled3]}]
    unset scaler1 scaler2 scaler3 tensor1 tensor2 tensor3 scaled1 scaled2 scaled3
    set result
} {1}

cleanupTests 