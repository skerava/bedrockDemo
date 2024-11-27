import boto3
import tool_use_print_utils as output
import logging
import os
import importlib

os.chdir("AIGC/basicClass")

logging.basicConfig(level=logging.INFO, format="%(message)s")

"""
定义一个BedrockConfig类, 包含在调用Bedrock的时候的一些配置信息
"""

class BedrockConfig:
    session = boto3.Session(profile_name="skydev")
    client = session.client(service_name="bedrock-runtime", region_name="us-west-2")
    inference_config = {"maxTokens": 4096, "temperature": 0.9, "topP": 0.8}
    MAX_RECURSIONS = 5

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

    def run(self):
        """
        Starts the conversation with the user and handles the interaction with Bedrock.
        """
        
        # Print the greeting and a short user guide
        output.header()
        

        # Start with an emtpy conversation
        conversation = []

        # Get the first user input
        user_input = self._get_user_input()

        while user_input is not None:
            # Create a new message with the user input and append it to the conversation
            message = {"role": "user", "content": [{"text": user_input}]}
            conversation.append(message)

            # Send the conversation to Amazon Bedrock
            bedrock_response = self._send_conversation_to_bedrock(conversation)
            logging.info(f"Bedrock response: {bedrock_response}")

            # Recursively handle the model's response until the model has returned
            # its final response or the recursion counter has reached 0
            self._process_model_response(
                bedrock_response, conversation, max_recursion=BedrockConfig.MAX_RECURSIONS
            )
            logging.info(f"Conversation: {conversation}")

            # Repeat the loop until the user decides to exit the application
            user_input = self._get_user_input()
            
        output.footer()

    def _send_conversation_to_bedrock(self, conversation):
        """
        Sends the conversation, the system prompt, and the tool spec to Amazon Bedrock, and returns the response.

        :param conversation: The conversation history including the next message to send.
        :return: The response from Amazon Bedrock.
        """
        output.call_to_bedrock(conversation)

        # Send the conversation, system prompt, and tool configuration, and return the response
        return BedrockConfig.client.converse(
            modelId=self.model_id,
            messages=conversation,
            system=self.system_prompt,
            toolConfig=self.tool_config,
        )

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

        if max_recursion <= 0:
            # Stop the process, the number of recursive calls could indicate an infinite loop
            logging.warning(
                "Warning: Maximum number of recursions reached. Please try again."
            )
            exit(1)

        # Append the model's response to the ongoing conversation
        message = model_response["output"]["message"]
        conversation.append(message)

        if model_response["stopReason"] == "tool_use":
            # If the stop reason is "tool_use", forward everything to the tool use handler
            self._handle_tool_use(message, conversation, max_recursion)

        if model_response["stopReason"] == "end_turn":
            # If the stop reason is "end_turn", print the model's response text, and finish the process
            output.model_response(message["content"][0]["text"])
            return

    def _handle_tool_use(
        self, model_response, conversation, max_recursion=BedrockConfig.MAX_RECURSIONS
    ):

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
                            "content": [{"json": tool_response["content"]}],
                        }
                    }
                )

        # Embed the tool results in a new user message
        message = {"role": "user", "content": tool_results}

        # Append the new message to the ongoing conversation
        conversation.append(message)

        # Send the conversation to Amazon Bedrock
        response = self._send_conversation_to_bedrock(conversation)

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
        user_input = input(f"{prompt} (x to exit): ")

        if user_input == "":
            prompt = "Please enter your prompt"
            return ToolUseDemo._get_user_input(prompt)

        elif user_input.lower() == "x":
            return None

        else:
            return user_input
        
    def _load_tool_configs(self):
        tool_configs = []
        tools_directory = 'tools'
        if not os.path.exists(tools_directory):
            logging.error(f"Tools directory '{tools_directory}' does not exist.")
            return tool_configs
        for filename in os.listdir(tools_directory):
            if filename.endswith('.py'):
                tool_name = filename[:-3]  # Remove the .py extension
                try:
                    tool_module = importlib.import_module(f'{tools_directory}.{tool_name}')
                    tool_config_method = getattr(tool_module, 'tool_config')
                    tool_config = tool_config_method()
                    tool_configs.append(tool_config)
                except (ModuleNotFoundError, AttributeError) as e:
                    logging.error(f"Error loading tool config from {tool_name}: {e}")
        return tool_configs if tool_configs else None

def main():
    system_prompt = input("请输入system_prompt: ")
    tool_use_demo = ToolUseDemo(system_prompt, model_id="anthropic.claude-3-5-sonnet-20240620-v1:0")
    tool_use_demo.run()
    
if __name__ == "__main__":
    main()

