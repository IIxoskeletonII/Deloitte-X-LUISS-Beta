import tokenize
import io
import re

def remove_comments_and_emojis(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        source_code = f.read()
        
    # Remove emojis
    emoji_pattern = re.compile(
        r'['
        r'\U0001F300-\U0001F5FF'
        r'\U0001F600-\U0001F64F'
        r'\U0001F680-\U0001F6FF'
        r'\U0001F700-\U0001F77F'
        r'\U0001F780-\U0001F7FF'
        r'\U0001F800-\U0001F8FF'
        r'\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F'
        r'\U0001FA70-\U0001FAFF'
        r'\U00002702-\U000027B0'
        r'\U000024C2-\U0001F251'
        r'─' # Custom extra symbol often used linearly
        r']+', flags=re.UNICODE)
    
    source_code = emoji_pattern.sub('', source_code)
    
    # Remove comments
    io_obj = io.StringIO(source_code)
    out = ""
    last_lineno = -1
    last_col = 0
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            
            if start_line > last_lineno:
                last_col = 0
            
            # preserve spacing
            if last_col < start_col:
                out += " " * (start_col - last_col)
                
            if token_type == tokenize.COMMENT:
                pass
            else:
                out += token_string
                
            last_col = end_col
            last_lineno = end_line
    except tokenize.TokenError:
        out = source_code # fallback
        
    # Remove excessive blank lines
    cleaned_lines = []
    for line in out.split('\n'):
        if not line.strip():
            if cleaned_lines and not cleaned_lines[-1].strip():
                continue # skip multiple blank lines
            if not cleaned_lines:
                continue # skip leading blank lines
            cleaned_lines.append('')
        else:
            cleaned_lines.append(line.rstrip())
            
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines) + '\n')

remove_comments_and_emojis('analysis.py')
