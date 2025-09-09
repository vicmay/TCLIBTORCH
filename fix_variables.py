#!/usr/bin/env python3

import re

def fix_distributed_ops():
    """Fix distributed_operations.cpp"""
    filepath = "src/distributed_operations.cpp"
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the patterns
    fixes = [
        (r'// int dst = GetIntFromObj\(interp, objv\[2\]\);  // Unused variable', 
         'int dst = GetIntFromObj(interp, objv[2]);'),
        (r'// int src = GetIntFromObj\(interp, objv\[2\]\);  // Unused variable', 
         'int src = GetIntFromObj(interp, objv[2]);'),
        (r'// int tag = 0;  // Unused variable', 
         'int tag = 0;'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed distributed_operations.cpp")

def fix_advanced_signal():
    """Fix advanced_signal_processing.cpp"""
    filepath = "src/advanced_signal_processing.cpp"
    with open(filepath, 'r') as f:
        content = f.read()
    
    # These should already be fixed, but just in case
    fixes = [
        (r'// double sample_rate = GetDoubleFromObj\(interp, objv\[3\]\);  // Unused variable', 
         'double sample_rate = GetDoubleFromObj(interp, objv[3]);'),
        (r'// int dct_type = 2;  // Unused variable', 
         'int dct_type = 2;'),
    ]
    
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Fixed advanced_signal_processing.cpp")

if __name__ == "__main__":
    fix_distributed_ops()
    fix_advanced_signal() 