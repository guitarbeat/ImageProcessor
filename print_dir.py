import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime

def analyze_directory_structure(start_path='.'):
    # Comprehensive ignore lists
    ignore_dirs = {
        '.git', 'node_modules', '__pycache__', 'venv', '.venv',
        'env', '.env', '.idea', '.vscode', 'build', 'dist',
        'coverage', '.next', '.cache', '.husky', 'tmp', 'temp',
        'logs', '.terraform', '.serverless'
    }
    
    ignore_files = {
        # Config files
        '.DS_Store', 'Thumbs.db', '.gitignore', '.env',
        'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
        'requirements.txt', 'poetry.lock', 'Pipfile.lock',
        
        # Common config extensions
        '*.json', '*.config.js', '*.config.ts', '*.yaml', '*.yml',
        '*.toml', '*.ini', '*.cfg', '*.conf', '*.lock', '*.md',
        
        # Build/Cache files
        '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.class',
        '*.log', '*.pot', '*.mo', '*.map', '*.min.js', '*.min.css'
    }
    
    structure = []
    file_types = defaultdict(int)
    potential_issues = []
    
    def should_ignore_file(filename):
        # Check exact matches
        if filename in ignore_files:
            return True
        
        # Check pattern matches (*.extension)
        for pattern in ignore_files:
            if pattern.startswith('*'):
                if filename.endswith(pattern[1:]):
                    return True
        return False
    
    def print_tree(dir_path, prefix=""):
        try:
            entries = sorted(os.scandir(dir_path), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return
        
        for i, entry in enumerate(entries):
            if entry.name in ignore_dirs or entry.name.startswith('.'):
                continue
            
            if entry.is_file() and should_ignore_file(entry.name):
                continue
                
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            
            structure.append(f"{prefix}{connector}{entry.name}")
            
            if entry.is_file():
                ext = Path(entry.name).suffix
                if ext:
                    file_types[ext] += 1
                    
                if len(entry.name) > 50:
                    potential_issues.append(f"Long filename: {entry.name}")
                    
            if entry.is_dir():
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(entry.path, new_prefix)
    
    print_tree(start_path)
    
    # Create output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"directory_analysis_{timestamp}.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("📁 Project Structure:\n")
        f.write("\n".join(structure))
        
        f.write("\n\n📊 File Type Distribution:\n")
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{ext}: {count} files\n")
        
        if potential_issues:
            f.write("\n⚠️ Potential Improvements:\n")
            for issue in potential_issues:
                f.write(f"- {issue}\n")
            
            f.write("\nRecommendations:\n")
            f.write("- Consider grouping similar file types in dedicated directories\n")
            f.write("- Use meaningful, but concise filenames\n")
            f.write("- Avoid deep nesting (more than 4-5 levels)\n")
            f.write("- Keep related files close together\n")
    
    print(f"\nAnalysis saved to: {output_filename}")

if __name__ == "__main__":
    analyze_directory_structure()