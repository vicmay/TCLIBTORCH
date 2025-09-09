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

;# Test positional syntax
test sinh-1.1 {Basic positional syntax} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 1.1752011936438014) < 1e-6}
} 1

test sinh-1.2 {Positional syntax with tensor containing zero} {
    set t [torch::tensorCreate -data {0.0} -dtype float32]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {abs($value) < 1e-6}
} 1

test sinh-1.3 {Positional syntax with negative value} {
    set t [torch::tensorCreate -data {-1.0} -dtype float32]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {abs($value - (-1.1752011936438014)) < 1e-6}
} 1

test sinh-1.4 {Positional syntax with multiple values} {
    set t1 [torch::tensorCreate -data {0.0} -dtype float32]
    set t2 [torch::tensorCreate -data {1.0} -dtype float32]
    set t3 [torch::tensorCreate -data {-1.0} -dtype float32]
    
    set result1 [torch::sinh $t1]
    set result2 [torch::sinh $t2]
    set result3 [torch::sinh $t3]
    
    set val1 [torch::tensorItem $result1]
    set val2 [torch::tensorItem $result2]
    set val3 [torch::tensorItem $result3]
    
    expr {abs($val1) < 1e-6 && abs($val2 - 1.1752011936438014) < 1e-6 && abs($val3 - (-1.1752011936438014)) < 1e-6}
} 1

;# Test named parameter syntax
test sinh-2.1 {Named parameter syntax with -input} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    set result [torch::sinh -input $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 1.1752011936438014) < 1e-6}
} 1

test sinh-2.2 {Named parameter syntax with -tensor} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    set result [torch::sinh -tensor $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 1.1752011936438014) < 1e-6}
} 1

test sinh-2.3 {Named parameter with zero value} {
    set t [torch::tensorCreate -data {0.0} -dtype float32]
    set result [torch::sinh -input $t]
    set value [torch::tensorItem $result]
    expr {abs($value) < 1e-6}
} 1

test sinh-2.4 {Named parameter with multiple values} {
    set t1 [torch::tensorCreate -data {0.0} -dtype float32]
    set t2 [torch::tensorCreate -data {2.0} -dtype float32]
    
    set result1 [torch::sinh -input $t1]
    set result2 [torch::sinh -input $t2]
    
    set val1 [torch::tensorItem $result1]
    set val2 [torch::tensorItem $result2]
    
    expr {abs($val1) < 1e-6 && abs($val2 - 3.6268604078089807) < 1e-5}
} 1

;# Test camelCase alias
test sinh-3.1 {CamelCase alias basic test} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    set result [torch::siNh $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 1.1752011936438014) < 1e-6}
} 1

test sinh-3.2 {CamelCase alias with named parameters} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    set result [torch::siNh -input $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 1.1752011936438014) < 1e-6}
} 1

;# Test consistency between syntaxes
test sinh-4.1 {Consistency between positional and named syntax} {
    set t [torch::tensorCreate -data {1.5} -dtype float32]
    set result1 [torch::sinh $t]
    set result2 [torch::sinh -input $t]
    set value1 [torch::tensorItem $result1]
    set value2 [torch::tensorItem $result2]
    expr {abs($value1 - $value2) < 1e-8}
} 1

test sinh-4.2 {Consistency between original and camelCase} {
    set t [torch::tensorCreate -data {1.5} -dtype float32]
    set result1 [torch::sinh $t]
    set result2 [torch::siNh $t]
    set value1 [torch::tensorItem $result1]
    set value2 [torch::tensorItem $result2]
    expr {abs($value1 - $value2) < 1e-8}
} 1

test sinh-4.3 {Consistency between all syntax variations} {
    set t [torch::tensorCreate -data {2.0} -dtype float32]
    set result1 [torch::sinh $t]
    set result2 [torch::sinh -input $t]
    set result3 [torch::siNh $t]
    set result4 [torch::siNh -tensor $t]
    
    set value1 [torch::tensorItem $result1]
    set value2 [torch::tensorItem $result2]
    set value3 [torch::tensorItem $result3]
    set value4 [torch::tensorItem $result4]
    
    set consistent 1
    if {abs($value1 - $value2) >= 1e-8 || abs($value1 - $value3) >= 1e-8 || abs($value1 - $value4) >= 1e-8} {
        set consistent 0
    }
    set consistent
} 1

;# Test error handling
test sinh-5.1 {Error: No arguments} {
    catch {torch::sinh} msg
    string match "*Usage*" $msg
} 1

test sinh-5.2 {Error: Invalid tensor name} {
    catch {torch::sinh invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} 1

test sinh-5.3 {Error: Missing parameter value} {
    catch {torch::sinh -input} msg
    string match "*Missing value for parameter*" $msg
} 1

test sinh-5.4 {Error: Unknown parameter} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    catch {torch::sinh -unknown $t} msg
    string match "*Unknown parameter*" $msg
} 1

test sinh-5.5 {Error: Too many positional arguments} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    catch {torch::sinh $t extra_arg} msg
    string match "*Usage*" $msg
} 1

test sinh-5.6 {Error: Invalid tensor with named parameter} {
    catch {torch::sinh -input invalid_tensor} msg
    string match "*Invalid tensor name*" $msg
} 1

;# Test different data types
test sinh-6.1 {Float64 tensor} {
    set t [torch::tensorCreate -data {1.0} -dtype float64]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 1.175201) < 1e-5}
} 1

test sinh-6.2 {Float32 tensor} {
    set t [torch::tensorCreate -data {1.0} -dtype float32]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 1.1752011936438014) < 1e-6}
} 1

;# Test edge cases
test sinh-7.1 {Large positive value} {
    set t [torch::tensorCreate -data {5.0} -dtype float32]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {$value > 70.0 && $value < 80.0}
} 1

test sinh-7.2 {Large negative value} {
    set t [torch::tensorCreate -data {-5.0} -dtype float32]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {$value < -70.0 && $value > -80.0}
} 1

test sinh-7.3 {Small positive value} {
    set t [torch::tensorCreate -data {0.1} -dtype float32]
    set result [torch::sinh $t]
    set value [torch::tensorItem $result]
    expr {abs($value - 0.10016675) < 1e-6}
} 1

test sinh-7.4 {Multi-dimensional tensor} {
    set t [torch::tensorCreate -data {1.0 0.0 -1.0 2.0} -shape {2 2} -dtype float32]
    set result [torch::sinh $t]
    ;# Just verify it doesn't crash and returns a valid handle
    expr {[string length $result] > 0}
} 1

cleanupTests 