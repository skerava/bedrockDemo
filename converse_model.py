import boto3
import tool_use_print_utils as output
import logging
import os
import importlib
from botocore.exceptions import ClientError
from Quartz.CoreGraphics import CGDisplayBounds, CGMainDisplayID
import json
from config.ignore_tool_config import IgnoreTool

logging.basicConfig(
    filename='test.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    #filemode='a'
    )

"""
定义一个BedrockConfig类, 包含在调用Bedrock的时候的一些配置信息
"""
def get_screen_resolution():
    main_display_id = CGMainDisplayID()
    main_display_bounds = CGDisplayBounds(main_display_id)
    width = int(main_display_bounds.size.width)
    height = int(main_display_bounds.size.height)
    return width, height

class BedrockConfig:
    session = boto3.Session(profile_name="skydev") #更改为自己的profile
    client = session.client(service_name="bedrock-runtime", region_name="us-west-2")
    inference_config = {"maxTokens": 4096, "temperature": 0.9, "topP": 0.8}
    MAX_RECURSIONS = 10

class ToolUseDemo:
    """
    Demonstrates the tool use feature with the Amazon Bedrock Converse API.
    """

    def __init__(self, system_prompt, model_id):
        # Prepare the system prompt
        self.system_prompt = [{"text": system_prompt}]

        # Dynamically load tool configurations from the tools folder
        self.tool_config = {'tools':self._load_tool_configs()}
        
        self.model_id = model_id
        self.width, self.height = get_screen_resolution()
        print(f"Screen resolution: {self.width}x{self.height}")
        logging.info(f"Screen resolution: {self.width}x{self.height}")
        
        self.additional_model_request_fields={
            "tools": [
                {
                    "type": "computer_20241022",
                    "name": "computer",
                    "display_height_px": self.height,
                    "display_width_px": self.width,
                    "display_number": 0
                },
                {
                    "type": "bash_20241022",
                    "name": "bash",

                },
                {
                    "type": "text_editor_20241022",
                    "name": "str_replace_editor",
                }
            ],
            "anthropic_beta": ["computer-use-2024-10-22"]
        }
        
    def run(self):
        """
        Starts the conversation with the user and handles the interaction with Bedrock.
        """
        
        # Print the greeting and a short user guide
        output.header()
        logging.info("Starting the conversation with the user.")

        # Start with an emtpy conversation
        conversation = []
        
        user_input = self._get_user_input()
        image_message = self._load_image()

        while user_input is not None:
            logging.info(f"User input: {user_input}")
            # Create a new message with the user input and append it to the conversation
            content = []
            content.append({"text": user_input})
            if image_message is not None:
                content.append(image_message)
            message = {"role": "user", "content": content}
            conversation.append(message)

            # Send the conversation to Amazon Bedrock
            bedrock_response = self._send_conversation_to_bedrock(conversation)
            logging.info(f"Bedrock response: {bedrock_response}")
            #logging.info(f"Bedrock response: {bedrock_response}")

            # Recursively handle the model's response until the model has returned
            # its final response or the recursion counter has reached 0
            self._process_model_response(
                bedrock_response, conversation, max_recursion=BedrockConfig.MAX_RECURSIONS
            )
            #logging.info(f"Conversation: {conversation}")

            # Repeat the loop until the user decides to exit the application
            user_input = self._get_user_input()
            
        logging.info("User exited the conversation.")
        output.footer()

    def _send_conversation_to_bedrock(self, conversation):
        """
        Sends the conversation, the system prompt, and the tool spec to Amazon Bedrock, and returns the response.

        :param conversation: The conversation history including the next message to send.
        :return: The response from Amazon Bedrock.
        """
        output.call_to_bedrock(conversation)
        logging.info(f"Sending conversation to Bedrock:{conversation}")

        # Send the conversation, system prompt, and tool configuration, and return the response
        #logging.info(f"BedrockConfig.client.converse(\nmodelId={self.model_id},\nmessages={conversation},\nsystem={self.system_prompt},\ntoolConfig={self.tool_config},)")
        #logging.info(f"Additional model request fields: {additional_model_request_fields}")
        try:
            response = BedrockConfig.client.converse(
                    modelId=self.model_id,
                    messages=conversation,
                    system=self.system_prompt,
                    toolConfig=self.tool_config,
                    additionalModelRequestFields=self.additional_model_request_fields,
                )
            logging.info(f"Received response from Bedrock: {response}")
            return response
        except ClientError as e:
            logging.error(f"Error calling Bedrock: {e}")
        except Exception as e:
            logging.info(f"An unexpected error occurred: {e}")
            exit(1)


    def _process_model_response(
        self, model_response, conversation, max_recursion=BedrockConfig.MAX_RECURSIONS
    ):
        """
        Processes the response received via Amazon Bedrock and performs the necessary actions
        based on the stop reason.

        :param model_response: The model's response returned via Amazon Bedrock.
        :param conversation: The conversation history.
        :param max_recursion: The maximum number of recursive calls allowed.
        """
        logging.info(f"Processing model response: {model_response}")

        if max_recursion <= 0:
            # Stop the process, the number of recursive calls could indicate an infinite loop
            logging.warning(
                "Warning: Maximum number of recursions reached. Please try again."
            )
            exit(1)

        # Append the model's response to the ongoing conversation
        message = model_response["output"]["message"]
        conversation.append(message)
        #logging.info(f"Conversation: {conversation}")

        if model_response["stopReason"] == "tool_use":
            logging.info("Model requested tool use.")
            # If the stop reason is "tool_use", forward everything to the tool use handler
            self._handle_tool_use(message, conversation, max_recursion)

        if model_response["stopReason"] == "end_turn":
            logging.info("Model ended its turn.")
            # If the stop reason is "end_turn", print the model's response text, and finish the process
            output.model_response(message["content"][0]["text"])
            return

    def _handle_tool_use(
        self, model_response, conversation, max_recursion=BedrockConfig.MAX_RECURSIONS
    ):
        logging.info(f"Handling tool use: {model_response}")

        # Initialize an empty list of tool results
        tool_results = []

        # The model's response can consist of multiple content blocks
        for content_block in model_response["content"]:
            if "text" in content_block:
                # If the content block contains text, print it to the console
                output.model_response(content_block["text"])

            if "toolUse" in content_block:
                # If the content block is a tool use request, forward it to the tool
                tool_response = self._invoke_tool(content_block["toolUse"])

                # Add the tool use ID and the tool's response to the list of results
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": (tool_response["toolUseId"]),
                            "content": [tool_response["content"]],
                        }
                    }
                )

        # Embed the tool results in a new user message
        message = {"role": "user", "content": tool_results}
        logging.info(f"Tool results: {tool_results}")

        # Append the new message to the ongoing conversation
        conversation.append(message)
        #logging.info(f"Conversation: {conversation}")

        # Send the conversation to Amazon Bedrock
        response = self._send_conversation_to_bedrock(conversation)
        logging.info(f"Received response after tool use: {response}")

        # Recursively handle the model's response until the model has returned
        # its final response or the recursion counter has reached 0
        self._process_model_response(response, conversation, max_recursion - 1)

    def _invoke_tool(self, payload):
        """
        Invokes the specified tool with the given payload and returns the tool's response.
        If the requested tool does not exist, an error message is returned.

        :param payload: The payload containing the tool name and input data.
        :return: The tool's response or an error message.
        """
        logging.info(f"Invoking tool '{payload['name']}' with input data: {payload['input']}")
        #logging.info(f"Invoking tool '{payload['name']}' with input data: {payload['input']}")
        tool_name = payload["name"]
        input_data = payload["input"]
        output.tool_use(tool_name, input_data)
        
        """
        Dynamically imports and invokes the specified tool's invoke method.
        """
        try:
            # Dynamically import the tool module
            tool_module = importlib.import_module(f'tools.{tool_name}')
            
            # Get the invoke method from the tool module
            invoke_method = getattr(tool_module, 'invoke')
            
            # Call the invoke method and return the result
            response =  invoke_method(input_data)
            #logging.info(f"Tool '{tool_name}' invoked successfully.")
        
        except ModuleNotFoundError:
            error_message = (
                f"Tool '{tool_name}' not found in the tools directory."
            )
            logging.error(f"Tool '{tool_name}' not found in the tools directory.")
            response = {"error": "true", "message": error_message}
        except AttributeError:
            error_message = (
                f"Tool '{tool_name}' does not have an 'invoke' method."
            )
            logging.error(f"Tool '{tool_name}' does not have an 'invoke' method.")
            response = {"error": "true", "message": error_message}
            
        #logging.info(f"Tool response: {response}")
        return {"toolUseId": payload["toolUseId"], "content": response}

    @staticmethod
    def _get_user_input(prompt = "Please enter your prompt"):
        output.separator()
        logging.info(f"User input prompt: {prompt}")
        user_input = input(f"{prompt} (x to exit): ")

        if user_input == "":
            prompt = "Please enter your prompt"
            return ToolUseDemo._get_user_input(prompt)

        elif user_input.lower() == "x":
            return None

        else:
            return user_input

    @staticmethod
    def _whether_to_use_computer_use(prompt="Whether to use Computer Use?"):
        """_summary_

        Args:
            prompt (str, optional): _description_. Defaults to "Whether to use Computer Use?".

        Returns:
            _type_: _description_
        """
        output.separator()
        logging.info(f"User decision prompt: {prompt}")
        user_input = input(f"{prompt} (y/n): ")

        if user_input.lower() == "y":
            return True

        elif user_input.lower() == "n":
            return False

        else:
            return ToolUseDemo._whether_to_use_computer_use(prompt)
    
    @staticmethod
    def _load_image(prompt="Input the image path"):
        output.separator()
        logging.info(f"Image path prompt: {prompt}")
        image_path = input(f"{prompt} (x to exit): ")

        if image_path == "":
            prompt = "Input the image path"
            return ToolUseDemo._load_image(prompt)

        elif image_path.lower() == "x":
            return None

        else:
            try:
                with open(image_path, "rb") as image_file:
                    image_bytes = image_file.read()
                return image_bytes
            except FileNotFoundError:
                logging.error(f"File not found at: {image_path}")
                prompt = f"File not found at: {image_path}.Input the image path again"
                return ToolUseDemo._load_image(prompt)
    

        
    def _load_tool_configs(self):
        tool_configs = []
        tools_directory = 'tools'
        logging.info("Loading tool configurations.")
        if not os.path.exists(tools_directory):
            logging.error(f"Tools directory '{tools_directory}' does not exist.")
            return tool_configs
        for filename in os.listdir(tools_directory):
            if filename.endswith('.py') and filename not in IgnoreTool.TOOL_LIST:
                tool_name = filename[:-3]  # Remove the .py extension
                try:
                    tool_module = importlib.import_module(f'{tools_directory}.{tool_name}')
                    tool_config_method = getattr(tool_module, 'tool_config')
                    tool_config = tool_config_method()
                    tool_configs.append(tool_config)
                except (ModuleNotFoundError, AttributeError) as e:
                    logging.error(f"Error loading tool config from {tool_name}: {e}")
        logging.info(f"Loaded tool configurations: {tool_configs}")
        return tool_configs if tool_configs else None
    
class GenerateContent:

    def __init__(self, system_message, model_id):
        self.system_message = [{"text": system_message}]
        self.model_id = model_id

    def generate_content(
        self,
        user_message,
        client=BedrockConfig.client,
        inference_config=BedrockConfig.inference_config,
    ):
        self.conversation = [
            {"role": "user", "content": [{"text": user_message}]},
            {"role": "assistant", "content": [{"text": "按照要求，输出如下:"}]},
        ]
        logging.info(f"Generating content for user message: {user_message}")
        try:
            for attempt in range(3):
                response = client.converse(
                    modelId=self.model_id,
                    system=self.system_message,
                    messages=self.conversation,
                    inferenceConfig=inference_config,
                )
                response_text = response["output"]["message"]["content"][0]["text"]
                text_json = self.try_json_parse(response_text)
                if text_json is not None:
                    return text_json
                logging.warning(f"Attempt {attempt + 1} failed, retrying...")
            raise ValueError("Failed to generate valid response after 3 attempts")
        except (ClientError, Exception) as e:
            logging.error(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            exit(1)

    @staticmethod
    def read_file(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
        logging.info(f"Reading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
        
        # 写一个静态方法，将输入尝试转化成json，失败则返回None
    @staticmethod
    def try_json_parse(input):
        logging.info("Attempting to parse JSON.")
        try:
            return json.loads(input)
        except json.JSONDecodeError:
            return None


def main():
    system_prompt = input("请输入system_prompt: ")
    logging.info("Starting main function.")
    tool_use_demo = ToolUseDemo(system_prompt, model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
    tool_use_demo.run()
    logging.info("Main function completed.")
    
if __name__ == "__main__":
    main()

