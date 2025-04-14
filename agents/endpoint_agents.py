import json
from smolagents import ToolCallingAgent, CodeAgent, LiteLLMModel, UserInputTool
from models.schemas import EndpointSchema, ParameterSchema
from tools.search_tools import semantic_openapi_search, validate_endpoint_format
from tools.rest_api_tool import execute_rest_api_request

user_input_tool = UserInputTool()

def create_endpoint_retriever_agent(model):
    """Create and configure the endpoint retriever agent"""
    endpoint_retriever_agent = ToolCallingAgent(
        tools=[semantic_openapi_search, validate_endpoint_format],
        model=model,
        max_steps=15,
        name="endpoint_retreiever_agent",
        description="This is an agent that gets appropriate endpoint to make the request to. Provide it user query and the file to be searched as one parameter in natural language.",
    )
    
     # Customize the agent's prompt templates
    endpoint_retriever_agent.prompt_templates["managed_agent"]["task"] = (
    endpoint_retriever_agent.prompt_templates["managed_agent"]["task"] + 
    ".\nvalidate_endpoint_format tool expects dictionary to adhere to EndpointSchema: " + 
    json.dumps(EndpointSchema.model_json_schema(), indent=2)
    ) + ".\n Return the response of validate_endpoint_format as final answer once successful."
    
    endpoint_retriever_agent.prompt_templates["final_answer"]["pre_messages"] = (
    endpoint_retriever_agent.prompt_templates["final_answer"]["pre_messages"] + 
    ".\nYour answer should adhere to EndpointSchema: " + 
    json.dumps(EndpointSchema.model_json_schema(), indent=2)
    )
    
    return endpoint_retriever_agent

def create_request_executor_agent(model):
    request_executor_agent = ToolCallingAgent(
        tools=[execute_rest_api_request, semantic_openapi_search, user_input_tool],
        model=model,
        max_steps=20,
        name="request_executor_agent",
        description=f"This agent executes HTTP requests against the provided endpoint. Provide it file name, any special user query information and api information in the format:\n {json.dumps(EndpointSchema.model_json_schema(), indent=2)}" +
        "as a string."
    )

    request_executor_agent.prompt_templates["managed_agent"]["task"] = (
    request_executor_agent.prompt_templates["managed_agent"]["task"] + 
    "\nYou may query the user using user_input_tool if more information is required to make the request but refrain if unnecessary. If no information about the optional fields is given, leave them as is." +
    "\nUse semantic_openapi_search to search for more examples or schema examples if not enough provided." +
    "\nEndpointSchema:\n" + json.dumps(EndpointSchema.model_json_schema(), indent=2)
    )
    return request_executor_agent

def create_manager_agent(model, endpoint_retriever_agent, request_executor_agent, docs_path_to_endpoint_summary):
    """Create and configure the manager agent"""    
    manager_agent = CodeAgent(
        tools=[user_input_tool],
        model=model,
        managed_agents=[endpoint_retriever_agent, request_executor_agent],
    )
    
    # Customize the manager agent's system prompt
    manager_agent.prompt_templates["system_prompt"] = manager_agent.prompt_templates["system_prompt"] + f"""
        Context:
        Help user find the endpoiint they need based on user query. Endpoints are described by files as below, 
        you will see the json files with their description:
        {docs_path_to_endpoint_summary}
        
        Based on this information identify which file is likely to contain the endpoint needed by user. Once you know the file, use endpoint_retreiever_agent
        it will help retrieve the specific endpoint details required. 
        
        Once identified, return as per request or make use of request_executor_agent by providing it response from endpoint_retreiever_agent, file name and any other information to 
        send request to api.
    """
    
    return manager_agent