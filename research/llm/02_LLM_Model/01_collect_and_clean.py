import os
import glob

def clean_code(file_path):
    """
    Step 3: Clean the data.
    Remove unnecessary comments, logs, or other clutter from the code.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Remove lines that start with a comment (#)
    clean_lines = [line for line in lines if not line.strip().startswith('#')]
    return ''.join(clean_lines)

def collect_files(directory, extensions=('.py', '.md')):
    """
    Step 2: Collect the relevant files.
    Gathers specific types of files from a directory for training data.
    """
    collected_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                collected_files.append(os.path.join(root, file))
    return collected_files

if __name__ == "__main__":
    # Example usage: collect and clean Python files from this directory
    current_dir = "."
    files = collect_files(current_dir, extensions=('.py',))
    
    output_filename = 'cleaned_dataset.txt'
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        for fpath in files:
            # We don't want to include the output file if we accidentally read it
            if fpath.endswith(output_filename):
                continue
            
            out_file.write(clean_code(fpath))
            out_file.write("\n\n")
            
    print(f"Collected and cleaned {len(files)} scripts into '{output_filename}'")
