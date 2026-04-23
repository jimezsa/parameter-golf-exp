import lzma
import base64
import argparse
import sys
import re

def minify_python(code):
    """Basic minification: remove comments and extra whitespace."""
    # Remove single line comments
    code = re.sub(r'(?m)^ *#.*$', '', code)
    code = re.sub(r'(?m) +#.*$', '', code)
    # Remove empty lines
    code = re.sub(r'\n\s*\n', '\n', code)
    return code.strip()

def pack_submission(input_script, output_script, minify=False):
    # 1. Read the original code
    with open(input_script, 'r') as f:
        code = f.read()
    
    original_size = len(code)
    
    if minify:
        try:
            import python_minifier
            code = python_minifier.minify(code)
        except ImportError:
            print("python-minifier not found. Falling back to basic minification.")
            code = minify_python(code)

    minified_size = len(code)

    # 2. Compress the code string directly using raw LZMA2 (preset 9 for max compression)
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9}]
    compressed_bytes = lzma.compress(code.encode('utf-8'), format=lzma.FORMAT_RAW, filters=filters)
    
    # 3. Base85 encode it (more space-efficient than Base64)
    b85_string = base64.b85encode(compressed_bytes).decode('utf-8')
    
    # 4. Create the final one-liner wrapper
    wrapper = (
        'import lzma as L,base64 as B\n'
        f'exec(L.decompress(B.b85decode("{b85_string}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}]))\n'
    )
    
    # 5. Write out the compressed training script
    with open(output_script, 'w') as f:
        f.write(wrapper)
        
    print(f"Original size: {original_size / 1024:.2f} KB")
    if minify:
        print(f"Minified size: {minified_size / 1024:.2f} KB")
    print(f"Packed size:   {len(wrapper) / 1024:.2f} KB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack a Python script for submission.')
    parser.add_argument('input', help='Input Python script')
    parser.add_argument('output', help='Output Python script')
    parser.add_argument('--minify', action='store_true', help='Minify code before packing (requires python-minifier)')
    args = parser.parse_args()
    
    pack_submission(args.input, args.output, args.minify)
