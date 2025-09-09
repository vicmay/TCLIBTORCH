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

# Test gru with positional syntax
test gru-1.1 {Basic positional syntax} {
    set layer [torch::gru 10 20]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-1.2 {Positional syntax with num_layers} {
    set layer [torch::gru 10 20 2]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-1.3 {Positional syntax with bias} {
    set layer [torch::gru 10 20 1 0]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-1.4 {Positional syntax with batch_first} {
    set layer [torch::gru 10 20 1 1 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-1.5 {Positional syntax with dropout} {
    set layer [torch::gru 10 20 2 1 0 0.2]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-1.6 {Positional syntax with bidirectional} {
    set layer [torch::gru 10 20 1 1 0 0.0 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

# Test gru with named parameter syntax
test gru-2.1 {Named parameter syntax - basic} {
    set layer [torch::gru -input_size 10 -hidden_size 20]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.2 {Named parameter syntax - camelCase parameters} {
    set layer [torch::gru -inputSize 10 -hiddenSize 20]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.3 {Named parameter syntax - with num_layers} {
    set layer [torch::gru -input_size 10 -hidden_size 20 -num_layers 2]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.4 {Named parameter syntax - with numLayers camelCase} {
    set layer [torch::gru -inputSize 10 -hiddenSize 20 -numLayers 2]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.5 {Named parameter syntax - with bias} {
    set layer [torch::gru -input_size 10 -hidden_size 20 -bias 0]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.6 {Named parameter syntax - with batch_first} {
    set layer [torch::gru -input_size 10 -hidden_size 20 -batch_first 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.7 {Named parameter syntax - with batchFirst camelCase} {
    set layer [torch::gru -inputSize 10 -hiddenSize 20 -batchFirst 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.8 {Named parameter syntax - with dropout} {
    set layer [torch::gru -input_size 10 -hidden_size 20 -num_layers 2 -dropout 0.2]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.9 {Named parameter syntax - with bidirectional} {
    set layer [torch::gru -input_size 10 -hidden_size 20 -bidirectional 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-2.10 {Named parameter syntax - all parameters} {
    set layer [torch::gru -inputSize 10 -hiddenSize 20 -numLayers 2 -bias 1 -batchFirst 1 -dropout 0.1 -bidirectional 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

# Test camelCase alias
test gru-3.1 {camelCase alias - basic} {
    set layer [torch::Gru -input_size 10 -hidden_size 20]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-3.2 {camelCase alias - with camelCase parameters} {
    set layer [torch::Gru -inputSize 10 -hiddenSize 20 -numLayers 2]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-3.3 {camelCase alias - positional syntax} {
    set layer [torch::Gru 10 20 2 1 1 0.1 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-3.4 {camelCase alias - mixed parameters} {
    set layer [torch::Gru -inputSize 10 -hidden_size 20 -numLayers 2 -batch_first 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

# Test layer functionality
test gru-4.1 {Layer functionality - basic creation} {
    set layer [torch::gru 16 32]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-4.2 {Layer functionality - different configurations} {
    set layer [torch::gru 32 64 2]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-4.3 {Layer functionality - with dropout} {
    set layer [torch::gru 16 32 3 1 0 0.5]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-4.4 {Layer functionality - bidirectional} {
    set layer [torch::gru -inputSize 16 -hiddenSize 32 -bidirectional 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

# Test syntax consistency
test gru-5.1 {Syntax consistency - positional vs named} {
    set layer1 [torch::gru 10 20 2 1 1 0.1 1]
    set layer2 [torch::gru -input_size 10 -hidden_size 20 -num_layers 2 -bias 1 -batch_first 1 -dropout 0.1 -bidirectional 1]
    
    # Both should succeed
    set success1 [expr {[string match "gru*" $layer1] && [string length $layer1] > 0}]
    set success2 [expr {[string match "gru*" $layer2] && [string length $layer2] > 0}]
    
    expr {$success1 && $success2}
} {1}

test gru-5.2 {Syntax consistency - camelCase alias} {
    set layer1 [torch::gru -input_size 10 -hidden_size 20]
    set layer2 [torch::Gru -input_size 10 -hidden_size 20]
    
    # Both should succeed
    set success1 [expr {[string match "gru*" $layer1] && [string length $layer1] > 0}]
    set success2 [expr {[string match "gru*" $layer2] && [string length $layer2] > 0}]
    
    expr {$success1 && $success2}
} {1}

test gru-5.3 {Syntax consistency - parameter alternatives} {
    set layer1 [torch::gru -inputSize 10 -hiddenSize 20 -numLayers 2]
    set layer2 [torch::gru -input_size 10 -hidden_size 20 -num_layers 2]
    
    # Both should succeed
    set success1 [expr {[string match "gru*" $layer1] && [string length $layer1] > 0}]
    set success2 [expr {[string match "gru*" $layer2] && [string length $layer2] > 0}]
    
    expr {$success1 && $success2}
} {1}

test gru-5.4 {Syntax consistency - batchFirst alternatives} {
    set layer1 [torch::gru -inputSize 10 -hiddenSize 20 -batchFirst 1]
    set layer2 [torch::gru -input_size 10 -hidden_size 20 -batch_first 1]
    
    # Both should succeed
    set success1 [expr {[string match "gru*" $layer1] && [string length $layer1] > 0}]
    set success2 [expr {[string match "gru*" $layer2] && [string length $layer2] > 0}]
    
    expr {$success1 && $success2}
} {1}

# Test error handling
test gru-6.1 {Error handling - no parameters} {
    catch {torch::gru} msg
    expr {[string match "*parameters*" $msg] || [string match "*invalid*" $msg]}
} {1}

test gru-6.2 {Error handling - insufficient parameters} {
    catch {torch::gru 10} msg
    expr {[string match "*Usage*" $msg] || [string match "*Wrong*" $msg]}
} {1}

test gru-6.3 {Error handling - invalid input_size} {
    catch {torch::gru -input_size -1 -hidden_size 20} msg
    expr {[string match "*invalid*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test gru-6.4 {Error handling - invalid hidden_size} {
    catch {torch::gru -input_size 10 -hidden_size 0} msg
    expr {[string match "*invalid*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test gru-6.5 {Error handling - invalid num_layers} {
    catch {torch::gru -input_size 10 -hidden_size 20 -num_layers 0} msg
    expr {[string match "*invalid*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test gru-6.6 {Error handling - invalid dropout} {
    catch {torch::gru -input_size 10 -hidden_size 20 -dropout -0.1} msg
    expr {[string match "*invalid*" $msg] || [string match "*Invalid*" $msg]}
} {1}

test gru-6.7 {Error handling - unknown parameter} {
    catch {torch::gru -input_size 10 -hidden_size 20 -unknown_param value} msg
    expr {[string match "*Unknown*" $msg] || [string match "*unknown*" $msg]}
} {1}

test gru-6.8 {Error handling - non-numeric input_size} {
    catch {torch::gru -input_size "not_a_number" -hidden_size 20} msg
    expr {[string match "*Invalid*" $msg] || [string match "*expected*" $msg]}
} {1}

test gru-6.9 {Error handling - non-numeric hidden_size} {
    catch {torch::gru -input_size 10 -hidden_size "not_a_number"} msg
    expr {[string match "*Invalid*" $msg] || [string match "*expected*" $msg]}
} {1}

test gru-6.10 {Error handling - non-boolean bias} {
    catch {torch::gru -input_size 10 -hidden_size 20 -bias "not_a_boolean"} msg
    expr {[string match "*Invalid*" $msg] || [string match "*expected*" $msg]}
} {1}

test gru-6.11 {Error handling - non-boolean batch_first} {
    catch {torch::gru -input_size 10 -hidden_size 20 -batch_first "not_a_boolean"} msg
    expr {[string match "*Invalid*" $msg] || [string match "*expected*" $msg]}
} {1}

test gru-6.12 {Error handling - non-boolean bidirectional} {
    catch {torch::gru -input_size 10 -hidden_size 20 -bidirectional "not_a_boolean"} msg
    expr {[string match "*Invalid*" $msg] || [string match "*expected*" $msg]}
} {1}

# Test different configurations
test gru-7.1 {Different configurations - single layer} {
    set layer [torch::gru -inputSize 8 -hiddenSize 16 -numLayers 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-7.2 {Different configurations - multiple layers} {
    set layer [torch::gru -inputSize 8 -hiddenSize 16 -numLayers 4]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-7.3 {Different configurations - no bias} {
    set layer [torch::gru -inputSize 8 -hiddenSize 16 -bias 0]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-7.4 {Different configurations - batch first} {
    set layer [torch::gru -inputSize 8 -hiddenSize 16 -batchFirst 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

# Test edge cases
test gru-8.1 {Edge case - very small sizes} {
    set layer [torch::gru 1 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-8.2 {Edge case - large sizes} {
    set layer [torch::gru 512 1024]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-8.3 {Edge case - zero dropout} {
    set layer [torch::gru -inputSize 10 -hiddenSize 20 -numLayers 2 -dropout 0.0]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-8.4 {Edge case - maximum dropout} {
    set layer [torch::gru -inputSize 10 -hiddenSize 20 -numLayers 2 -dropout 1.0]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-8.5 {Edge case - typical speech recognition configuration} {
    set layer [torch::gru -inputSize 40 -hiddenSize 128 -numLayers 3 -dropout 0.3 -bidirectional 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

test gru-8.6 {Edge case - typical NLP configuration} {
    set layer [torch::gru -inputSize 300 -hiddenSize 256 -numLayers 2 -dropout 0.5 -batchFirst 1]
    set layer_valid [expr {[string match "gru*" $layer] && [string length $layer] > 0}]
    expr {$layer_valid}
} {1}

cleanupTests 