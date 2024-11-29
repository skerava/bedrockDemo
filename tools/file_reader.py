def tool_config():
    return {
        "toolSpec": {
            "name": "file_reader",
            "description": "Read content from a specified file path on local system",
            "inputSchema": {
                "json": {
                    "type": "object", 
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file that needs to be read"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    }

def invoke(input_data):
    """
    Reads and returns content from a file at the specified path.
    Returns the file content or an error message if the operation fails.

    :param input_data: The input data containing the file path.
    :return: The file content or an error message.
    """
    file_path = input_data.get("file_path")

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return {
                "status": "success",
                "content": content
            }
            
    except FileNotFoundError:
        return {
            "error": "FileNotFoundError",
            "message": f"File not found at path: {file_path}"
        }
    except PermissionError:
        return {
            "error": "PermissionError", 
            "message": f"Permission denied to read file: {file_path}"
        }
    except Exception as e:
        return {
            "error": type(e).__name__,
            "message": str(e)
        }
