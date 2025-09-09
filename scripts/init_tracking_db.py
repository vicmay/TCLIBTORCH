#!/usr/bin/env python3
"""
Initialize SQLite database for tracking LibTorch TCL command refactoring progress.
This replaces the markdown-based tracking system with a more reliable database approach.
"""

import sqlite3
import os
import sys
import subprocess
import re
from pathlib import Path

def create_database(db_path):
    """Create the tracking database with the required schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create commands table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT,
            source_file TEXT,
            has_dual_syntax BOOLEAN DEFAULT FALSE,
            has_camel_case_alias BOOLEAN DEFAULT FALSE,
            has_tests BOOLEAN DEFAULT FALSE,
            has_documentation BOOLEAN DEFAULT FALSE,
            is_complete BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create refactoring_log table for tracking changes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS refactoring_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command_name TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (command_name) REFERENCES commands (name)
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_name ON commands (name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_category ON commands (category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_commands_complete ON commands (is_complete)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_log_command ON refactoring_log (command_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_log_timestamp ON refactoring_log (timestamp)')
    
    conn.commit()
    return conn

def scan_source_files(src_dir):
    """Scan source files to find all torch:: commands."""
    commands = {}
    
    # Common patterns for torch commands
    patterns = [
        r'Tcl_CreateObjCommand\(interp,\s*"(torch::\w+)"',  # Command registration
        r'int\s+(\w+)_Cmd\(',  # Command function definitions
    ]
    
    src_path = Path(src_dir)
    for cpp_file in src_path.glob('*.cpp'):
        with open(cpp_file, 'r') as f:
            content = f.read()
            
        # Find torch:: command registrations
        for match in re.finditer(patterns[0], content):
            command_name = match.group(1)
            if command_name.startswith('torch::'):
                commands[command_name] = {
                    'source_file': str(cpp_file.relative_to(src_path.parent)),
                    'category': categorize_command(command_name, cpp_file.name)
                }
    
    return commands

def categorize_command(command_name, source_file):
    """Categorize commands based on their name and source file."""
    name = command_name.replace('torch::', '')
    
    if 'tensor_creation' in source_file or any(x in name for x in ['zeros', 'ones', 'empty', 'rand', 'full', 'eye', 'arange']):
        return 'tensor_creation'
    elif 'basic_tensor_ops' in source_file or any(x in name for x in ['add', 'mul', 'div', 'sub', 'matmul', 'bmm']):
        return 'basic_operations'
    elif 'basic_layers' in source_file or any(x in name for x in ['linear', 'conv', 'pool', 'dropout', 'batch']):
        return 'neural_layers'
    elif any(x in name for x in ['relu', 'sigmoid', 'tanh', 'softmax', 'log', 'exp', 'sqrt']):
        return 'activation_functions'
    elif any(x in name for x in ['sgd', 'adam', 'optimizer', 'scheduler']):
        return 'optimizers'
    elif any(x in name for x in ['loss', 'mse', 'cross_entropy', 'nll']):
        return 'loss_functions'
    else:
        return 'other'

def check_dual_syntax(command_name, src_dir):
    """Check if a command has dual syntax support by looking for parser functions."""
    # Look for ParseXXXArgs functions
    search_pattern = f"Parse.*Args.*{command_name.replace('torch::', '').replace('_', '').lower()}"
    
    src_path = Path(src_dir)
    for cpp_file in src_path.glob('*.cpp'):
        with open(cpp_file, 'r') as f:
            content = f.read()
            
        # Look for dual syntax patterns
        if ('ParseEmptyLikeArgs' in content and any(x in command_name for x in ['like'])) or \
           (f'Parse{command_name.replace("torch::", "").title().replace("_", "")}Args' in content) or \
           ('Named parameter syntax' in content and command_name in content):
            return True
    
    return False

def check_camel_case_alias(command_name, src_dir):
    """Check if a command has a camelCase alias registered."""
    src_path = Path(src_dir) / 'libtorchtcl.cpp'
    if not src_path.exists():
        return False
        
    with open(src_path, 'r') as f:
        content = f.read()
    
    # Look for camelCase registration patterns
    base_name = command_name.replace('torch::', '')
    camel_patterns = [
        f'torch::{to_camel_case(base_name)}',
        f'{base_name}Layer',
        f'{base_name}Function'
    ]
    
    for pattern in camel_patterns:
        if pattern in content:
            return True
    
    return False

def to_camel_case(snake_str):
    """Convert snake_case to camelCase."""
    components = snake_str.split('_')
    return components[0] + ''.join(word.capitalize() for word in components[1:])

def check_tests(command_name):
    """Check if tests exist for the command."""
    test_file = f"tests/refactored/{command_name.replace('torch::', '')}_test.tcl"
    return os.path.exists(test_file)

def check_documentation(command_name):
    """Check if documentation exists for the command."""
    doc_file = f"docs/refactored/{command_name.replace('torch::', '')}.md"
    return os.path.exists(doc_file)

def populate_database(conn, src_dir):
    """Populate the database with current command status."""
    cursor = conn.cursor()
    
    # Scan for commands
    commands = scan_source_files(src_dir)
    
    for command_name, info in commands.items():
        # Check current status
        has_dual_syntax = check_dual_syntax(command_name, src_dir)
        has_camel_case = check_camel_case_alias(command_name, src_dir)
        has_tests = check_tests(command_name)
        has_docs = check_documentation(command_name)
        is_complete = has_dual_syntax and has_camel_case and has_tests and has_docs
        
        # Insert or update command
        cursor.execute('''
            INSERT OR REPLACE INTO commands 
            (name, category, source_file, has_dual_syntax, has_camel_case_alias, 
             has_tests, has_documentation, is_complete, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (command_name, info['category'], info['source_file'], 
              has_dual_syntax, has_camel_case, has_tests, has_docs, is_complete))
        
        # Log the scan
        cursor.execute('''
            INSERT INTO refactoring_log (command_name, action, details)
            VALUES (?, ?, ?)
        ''', (command_name, 'scanned', f'Status: dual_syntax={has_dual_syntax}, camel={has_camel_case}, tests={has_tests}, docs={has_docs}'))
    
    conn.commit()
    print(f"Populated database with {len(commands)} commands")

def main():
    """Main function to initialize the tracking database."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Usage: python3 init_tracking_db.py [database_path]")
        print("Initialize SQLite database for tracking LibTorch TCL command refactoring.")
        print("Default database path: ./refactoring_progress.db")
        return
    
    # Database path
    db_path = sys.argv[1] if len(sys.argv) > 1 else './refactoring_progress.db'
    src_dir = './src'
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory '{src_dir}' not found")
        sys.exit(1)
    
    print(f"Initializing refactoring tracking database: {db_path}")
    
    # Create and populate database
    conn = create_database(db_path)
    populate_database(conn, src_dir)
    
    # Print summary
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM commands')
    total = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM commands WHERE is_complete = 1')
    complete = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM commands WHERE has_dual_syntax = 1')
    dual_syntax = cursor.fetchone()[0]
    
    print(f"\nDatabase initialized successfully!")
    print(f"Total commands: {total}")
    print(f"Complete commands: {complete} ({complete/total*100:.1f}%)")
    print(f"Commands with dual syntax: {dual_syntax} ({dual_syntax/total*100:.1f}%)")
    
    conn.close()

if __name__ == '__main__':
    main() 