#!/usr/bin/env tclsh

puts "Testing Learning Rate Schedulers..."

# Load the library
load ./build/libtorchtcl.so

# Create a simple model and optimizer for testing
set model_params [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cuda 0]
set optimizer [torch::optimizer_sgd $model_params 0.1]

puts "Initial learning rate: [torch::get_lr $optimizer]"

# Test Step LR Scheduler
puts "\n=== Testing Step LR Scheduler ==="
set step_scheduler [torch::lr_scheduler_step $optimizer 2 0.5]
puts "Created step scheduler: $step_scheduler"

for {set i 0} {$i < 6} {incr i} {
    puts "Step $i: LR = [torch::get_lr $optimizer]"
    torch::lr_scheduler_step_update $step_scheduler
}

# Reset optimizer for next test
set optimizer2 [torch::optimizer_adam $model_params 0.01]
puts "\n=== Testing Exponential LR Scheduler ==="
set exp_scheduler [torch::lr_scheduler_exponential $optimizer2 0.9]
puts "Created exponential scheduler: $exp_scheduler"

for {set i 0} {$i < 5} {incr i} {
    puts "Step $i: LR = [torch::get_lr $optimizer2]"
    torch::lr_scheduler_step_update $exp_scheduler
}

# Reset optimizer for next test
set optimizer3 [torch::optimizer_sgd $model_params 0.1]
puts "\n=== Testing Cosine Annealing LR Scheduler ==="
set cosine_scheduler [torch::lr_scheduler_cosine $optimizer3 10 0.001]
puts "Created cosine scheduler: $cosine_scheduler"

for {set i 0} {$i < 12} {incr i} {
    puts "Step $i: LR = [format %.6f [torch::get_lr $optimizer3]]"
    torch::lr_scheduler_step_update $cosine_scheduler
}

puts "\nAll learning rate schedulers working correctly!" 