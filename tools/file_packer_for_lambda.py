import os
import zipfile
from requests.exceptions import RequestException



from converse_model import BedrockConfig
"""
system_prompt = '''You are an advanced AI security analyst and cloud deployment specialist. Your primary responsibility is to perform security analysis on provided code and manage AWS Lambda deployments accordingly.

SECURITY ANALYSIS PROTOCOL:
Analyze all code for malicious patterns including:
- Network attacks (DDoS, port scanning)
- Malware behaviors
- Unauthorized access attempts
- SQL/Command injection
- Cross-site scripting (XSS)
- Unauthorized scraping/crawling
- Privacy violations
- Credential theft
- System manipulation
- Other harmful patterns

EXECUTION RULES:
1. IF MALICIOUS CODE DETECTED:
   Output: {
     "status": "SECURITY_ALERT",
     "message": "⚠️ Malicious code detected",
     "details": {
       "patterns": [list of detected patterns],
       "code_sections": [affected code],
       "risks": [potential risks],
       "recommendations": [remediation steps]
     }
   }
   Then stop all operations.

2. IF CODE IS SAFE:
   Output: {
     "status": "SECURITY_PASS",
     "message": "✅ Security check passed"
   }
   Then execute:
   - Call file_packer_for_lambda
   - Package code for Lambda
   - Create new function
   - Return deployment status

IMPORTANT:
- Always complete security analysis first
- Never proceed with deployment if security concerns exist
- Provide clear, structured JSON responses
- Be thorough in security analysis
- Flag any potentially harmful patterns'''

"""

def tool_config():
    return {
        "toolSpec": {
            "name": "file_packer_for_lambda",
            "description": "Package files from a given path into a zip file and upload zip file to create a lambda function.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The file or directory path to package.",
                        },
                        "function_name": {
                            "type": "string",
                            "description": "The function name to upload the zip file and create.",
                        },
                    },
                    "required": ["path", "function_name"],
                }
            },
        }
    }

def invoke(input_data):
    """
    Packages files from the given path into a zip file and uploads it to the specified function_name.

    :param input_data: The input data containing the path and function_name.
    :return: The response from the upload function_name or an error message.
    """
    path = input_data.get("path")
    function_name = input_data.get("function_name")
    zip_name = f"{function_name}.zip"

    if not path or not function_name:
        return {"error": "Both 'path' and 'function_name' are required."}

    # Create zip file
    create_zip(zip_name, path)

    # Upload zip file
    try:
        response = create_lambda_function(zip_name, function_name)
        return response
    except RequestException as e:
        if e.response is not None:
            return e.response.json()
        else:
            return {"error": "Request failed", "message": str(e)}
    except Exception as e:
        return {"error": type(e).__name__, "message": str(e)}

# Create zip file
def create_zip(zip_name, path):
    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=path)
                        zipf.write(file_path, arcname=arcname)
            elif os.path.isfile(path):
                zipf.write(path, arcname=os.path.basename(path))
            else:
                return {"error": f"Path '{path}' does not exist."}
    except Exception as e:
        return {"error": f"Failed to create zip file: {str(e)}"}
    
def create_lambda_function(zip_file, function_name):
    """
    创建或更新 Lambda 函数
    :param zip_file: tar/zip 文件路径
    :param function_name: Lambda 函数名称(可选)
    """
    lambda_client = BedrockConfig.session.client(service_name='lambda')
    
    try:
        # 读取 zip 文件
        with open(zip_file, 'rb') as f:
            zip_bytes = f.read()
            
        # 尝试更新现有函数
        try:
            response = lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_bytes
            )
            print(f"Updated existing function: {function_name}")
            
        except lambda_client.exceptions.ResourceNotFoundException:
            # 如果函数不存在,创建新函数
            response = lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.11',  # 指定运行时
                Role='',  # 需要指定 IAM 角色
                Handler='security_handler.init',  # 处理程序
                Code={
                    'ZipFile': zip_bytes
                },
                Timeout=180,  # 超时时间(秒)
                MemorySize=128,  # 内存大小(MB)
                Environment={
                    "Variables": {
                        "ORIGINAL_HANDLER": f"{function_name}.lambda_handler"
                        }
                    },
                Layers= [
                    "",
                    ""
                    ]
            )
            print(f"Created new function: {function_name}")
            
        return {"json":response}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
