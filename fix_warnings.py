#!/usr/bin/env python3

import os
import re
import sys

def fix_unused_clientdata(filepath):
    """Fix unused ClientData parameter warnings in a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to match function signatures with ClientData parameter
    pattern = r'(int\s+\w+_Cmd\(ClientData\s+clientData,\s*Tcl_Interp\*\s*interp,\s*int\s+objc,\s*Tcl_Obj\*\s*const\s*objv\[\]\)\s*\{)'
    
    def replacement(match):
        func_signature = match.group(1)
        return func_signature + '\n    (void)clientData; // Suppress unused parameter warning'
    
    # Only add the suppression if it's not already there
    new_content = re.sub(pattern, replacement, content)
    
    # Also check if we need to add it after existing (void)clientData lines are already there
    # but make sure we don't duplicate
    lines = new_content.split('\n')
    modified_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        modified_lines.append(line)
        
        # Check if this line is a function signature with ClientData
        if re.match(r'int\s+\w+_Cmd\(ClientData\s+clientData,', line):
            # Look for the opening brace and add suppression after it
            if line.endswith('{'):
                # Check if next line already has the suppression
                if i + 1 < len(lines) and '(void)clientData' not in lines[i + 1]:
                    modified_lines.append('    (void)clientData; // Suppress unused parameter warning')
            elif i + 1 < len(lines) and lines[i + 1].strip() == '{':
                # Opening brace on next line
                modified_lines.append(lines[i + 1])  # Add the opening brace
                i += 1
                if i + 1 < len(lines) and '(void)clientData' not in lines[i + 1]:
                    modified_lines.append('    (void)clientData; // Suppress unused parameter warning')
        
        i += 1
    
    new_content = '\n'.join(modified_lines)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed unused ClientData warnings in {filepath}")
        return True
    return False

def fix_unused_variables(filepath):
    """Fix specific unused variable warnings."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Common patterns for unused variables that can be fixed
    fixes = [
        # Comment out unused variables with //
        (r'(\s+)(int\s+shape_arg_idx\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(bool\s+has_metric\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(bool\s+grad_averaging\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(bool\s+aligned\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(int\s+nhead\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(int\s+dct_type\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(double\s+sample_rate\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(int\s+dst\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(int\s+src\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
        (r'(\s+)(int\s+tag\s*=\s*[^;]+;)', r'\1// \2  // Unused variable'),
    ]
    
    modified = False
    for pattern, replacement in fixes:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            content = new_content
            modified = True
    
    if modified:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed unused variable warnings in {filepath}")
        return True
    return False

def main():
    src_dir = "src"
    if not os.path.exists(src_dir):
        print("Error: src directory not found")
        sys.exit(1)
    
    cpp_files = [f for f in os.listdir(src_dir) if f.endswith('.cpp')]
    
    total_fixed = 0
    for cpp_file in cpp_files:
        filepath = os.path.join(src_dir, cpp_file)
        print(f"Processing {filepath}...")
        
        fixed1 = fix_unused_clientdata(filepath)
        fixed2 = fix_unused_variables(filepath)
        
        if fixed1 or fixed2:
            total_fixed += 1
    
    print(f"\nFixed warnings in {total_fixed} files")

if __name__ == "__main__":
    main() 