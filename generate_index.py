import os
from pathlib import Path

def load_gitignore():
    """Load paths from .gitignore to exclude them in the index."""
    ignore_paths = set()
    if os.path.exists(".gitignore"):
        with open(".gitignore", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ignore_paths.add(line)
    return ignore_paths

def is_ignored(path, ignore_paths):
    """Check if a file or folder should be ignored based on .gitignore patterns."""
    for ignore_path in ignore_paths:
        if Path(ignore_path).match(path) or path.startswith(ignore_path):
            return True
    return False

def generate_index():
    base_dir = "."
    ignore_paths = load_gitignore()
    index_content = "# MLMI Notes\n\nWelcome to my notes for MLMI and related topics. Use the links below to navigate.\n\n---\n\n"

    for root, dirs, files in os.walk(base_dir):
        # Skip root directory
        if root == base_dir:
            continue
        
        # Skip any directories in .gitignore
        relative_root = os.path.relpath(root, base_dir)
        if is_ignored(relative_root, ignore_paths):
            continue

        section_name = os.path.basename(root)
        index_content += f"## {section_name.capitalize()}\n\n"

        for file in sorted(files):
            # Skip files that are in .gitignore or are not Markdown
            relative_file = os.path.join(relative_root, file)
            if is_ignored(relative_file, ignore_paths) or not file.endswith(".md"):
                continue

            # Format file path and name
            file_path = relative_file.replace(" ", "%20")
            file_name = os.path.splitext(file)[0].replace("-", " ").capitalize()
            index_content += f"- [{file_name}]({file_path})\n"
        index_content += "\n"

    with open("index.md", "w") as f:
        f.write(index_content)

if __name__ == "__main__":
    generate_index()
