import re  
from typing import List, Tuple, Dict


def parse_multipart_full(headers, rfile) -> Tuple[List[Tuple[str, bytes]], Dict[str, str]]:  
    """  
    Parses multipart/form-data without the cgi module (Python 3.13+ compatible).  
    Returns:  
            - files:  list of (filename, bytes)  
            - fields: dict of text fields e.g. {'chunk_mode': 'semantic'}  
    """  
    content_type = headers.get("Content-Type", "")  
    length       = int(headers.get("Content-Length", 0))  
    body         = rfile.read(length)

    # ── Extract boundary ──  
    boundary_match = re.search(r'boundary=(.+)', content_type)  
    if not boundary_match:  
        return [], {}

    boundary = boundary_match.group(1).strip().encode()  
    parts    = body.split(b'--' + boundary)

    files  = []  
    fields = {}

    for part in parts:  
        if b'Content-Disposition' not in part:  
            continue

        # Split headers from body  
        try:  
            header_section, data = part.split(b'\r\n\r\n', 1)  
        except ValueError:  
            continue

        data = data.rstrip(b'\r\n')

        if b'filename=' in header_section:  
            # ── File field ──  
            filename_match = re.search(  
                rb'filename=["\']?([^"\'\r\n;]+)["\']?',  
                header_section  
            )  
            if filename_match:  
                filename = filename_match.group(1).decode('utf-8', errors='replace').strip()  
                if filename and data:  
                    files.append((filename, data))  
                    print(f"[PARSER] File found: {filename} — {len(data)} bytes")  
        else:  
            # ── Text field ──  
            name_match = re.search(  
                rb'name=["\']?([^"\'\r\n;]+)["\']?',  
                header_section  
            )  
            if name_match:  
                field_name  = name_match.group(1).decode('utf-8', errors='replace').strip()  
                field_value = data.decode('utf-8', errors='replace').strip()  
                fields[field_name] = field_value  
                print(f"[PARSER] Field: {field_name} = {field_value}")

    return files, fields  
