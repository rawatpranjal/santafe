#!/usr/bin/env python3
"""
Script to aggregate all Python files and repository structure into a single text file.
This gives us a complete view of the codebase for analysis and planning.
"""

import os
from pathlib import Path
import datetime

def get_directory_tree(path, prefix="", ignore_dirs=None):
    """Generate a tree structure of directories and files."""
    if ignore_dirs is None:
        ignore_dirs = {'__pycache__', '.git', 'venv', 'htmlcov', '.pytest_cache', 'experiments'}
    
    tree_lines = []
    path = Path(path)
    
    # Get all items in directory
    items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        
        # Skip ignored directories
        if item.is_dir() and item.name in ignore_dirs:
            continue
            
        # Skip non-essential files
        if item.is_file() and not (item.suffix in ['.py', '.md', '.txt', '.sh'] or 
                                   item.name in ['requirements.txt', 'LICENSE', '.gitignore']):
            continue
        
        # Draw tree branch
        connector = "└── " if is_last else "├── "
        tree_lines.append(f"{prefix}{connector}{item.name}")
        
        # Recurse for directories
        if item.is_dir():
            extension = "    " if is_last else "│   "
            subtree = get_directory_tree(item, prefix + extension, ignore_dirs)
            tree_lines.extend(subtree)
    
    return tree_lines

def collect_python_files(root_path):
    """Collect all Python files in the repository (excluding tests)."""
    python_files = []
    ignore_dirs = {'__pycache__', '.git', 'venv', 'htmlcov', '.pytest_cache', 'tests', 'test_traders'}
    
    for root, dirs, files in os.walk(root_path):
        # Remove ignored directories from traversal
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        # Skip test directories and files
        if 'test' in root.lower():
            continue
            
        for file in files:
            # Skip test files
            if file.startswith('test_') or file.endswith('_test.py'):
                continue
                
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_path)
                
                # Focus on core src_code files
                if 'src_code' in rel_path or file in ['requirements.txt']:
                    python_files.append((rel_path, file_path))
    
    return sorted(python_files)

def main():
    # Set up paths
    repo_root = Path(__file__).parent.parent  # Go up from for_gemini to repo root
    output_file = Path(__file__).parent / "complete_codebase.txt"
    
    print(f"Scanning repository: {repo_root}")
    print(f"Output file: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("COMPLETE SANTAFE-1 CODEBASE AGGREGATION\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        # Write repository structure
        f.write("=" * 80 + "\n")
        f.write("REPOSITORY STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        
        tree = get_directory_tree(repo_root)
        for line in tree:
            f.write(line + "\n")
        
        # Collect and write all Python files
        python_files = collect_python_files(repo_root)
        
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write(f"PYTHON FILES ({len(python_files)} files)\n")
        f.write("=" * 80 + "\n")
        
        for rel_path, full_path in python_files:
            f.write(f"\n\n{'='*80}\n")
            f.write(f"FILE: {rel_path}\n")
            f.write(f"{'='*80}\n\n")
            
            try:
                with open(full_path, 'r', encoding='utf-8') as py_file:
                    content = py_file.read()
                    f.write(content)
            except Exception as e:
                f.write(f"ERROR READING FILE: {e}\n")
        
        # Add summary statistics
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Python files: {len(python_files)}\n")
        
        # Count lines of code
        total_lines = 0
        for _, full_path in python_files:
            try:
                with open(full_path, 'r') as py_file:
                    total_lines += len(py_file.readlines())
            except:
                pass
        
        f.write(f"Total lines of Python code: {total_lines:,}\n")
        
        # List main directories
        main_dirs = set()
        for rel_path, _ in python_files:
            parts = rel_path.split(os.sep)
            if len(parts) > 1:
                main_dirs.add(parts[0])
        
        f.write(f"Main directories with Python code: {', '.join(sorted(main_dirs))}\n")
    
    print(f"\n✅ Complete codebase aggregated to: {output_file}")
    print(f"   Total Python files: {len(python_files)}")
    print(f"   Total lines of code: {total_lines:,}")

if __name__ == "__main__":
    main()