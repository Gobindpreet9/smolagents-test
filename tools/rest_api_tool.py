import json
import os
import requests
from typing import Dict, Any, Optional, List, Union
from smolagents import tool
from models.schemas import EndpointSchema

# Note: Authentication is not handled in this version of the tool

@tool
def execute_rest_api_request(
    endpoint_data: str,
    base_url: str,
    path_params: Optional[Dict[str, str]] = None,
    query_params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    timeout: int = 30
) -> str:
    """
    Executes a REST API request based on the provided endpoint data and parameters.
    Authentication is not supported in the current version.
    
    Args:
        endpoint_data: A string representation of an EndpointSchema object (JSON or dict literal)
        base_url: The base URL for the API (e.g., 'https://api.example.com')
        path_params: Optional dictionary of path parameters to replace in the URL
        query_params: Optional dictionary of query parameters to add to the URL
        headers: Optional dictionary of HTTP headers (not for authentication)
        body: Optional request body (for POST, PUT methods) that can be a dictionary, list, or string
        timeout: Request timeout in seconds (default: 30)
    
    Returns:
        A string containing the response from the API, including status code, headers, and body
    """
    try:
        # Parse the endpoint data
        try:
            if endpoint_data.strip().startswith('{'):
                # Try to parse as JSON
                endpoint_dict = json.loads(endpoint_data)
            else:
                # Try to parse as Python dict literal
                import ast
                endpoint_dict = ast.literal_eval(endpoint_data)
                
            endpoint = EndpointSchema(**endpoint_dict)
        except Exception as e:
            return f"Error parsing endpoint data: {str(e)}. Expected format to match EndpointSchema."
        
        # Construct the full URL
        url = base_url.rstrip('/') + endpoint.url
        
        # Replace path parameters if provided
        if path_params:
            for param_name, param_value in path_params.items():
                placeholder = f"{{{param_name}}}"
                url = url.replace(placeholder, str(param_value))
        
        # Prepare request headers
        request_headers = headers or {}
        
        # Prepare request body
        request_body = None
        if body:
            if isinstance(body, str):
                try:
                    # Try to parse as JSON if it's a string
                    request_body = json.loads(body)
                except json.JSONDecodeError:
                    # If not valid JSON, use as raw string
                    request_body = body
            else:
                request_body = body
        
        # Execute the request based on the HTTP method
        method = endpoint.type.upper()
        response = None
        
        if method == 'GET':
            response = requests.get(
                url, 
                params=query_params, 
                headers=request_headers, 
                timeout=timeout
            )
        elif method == 'POST':
            response = requests.post(
                url, 
                params=query_params, 
                headers=request_headers, 
                json=request_body if isinstance(request_body, (dict, list)) else None,
                data=request_body if isinstance(request_body, str) else None,
                timeout=timeout
            )
        elif method == 'PUT':
            response = requests.put(
                url, 
                params=query_params, 
                headers=request_headers, 
                json=request_body if isinstance(request_body, (dict, list)) else None,
                data=request_body if isinstance(request_body, str) else None,
                timeout=timeout
            )
        elif method == 'DELETE':
            response = requests.delete(
                url, 
                params=query_params, 
                headers=request_headers, 
                json=request_body if isinstance(request_body, (dict, list)) else None,
                data=request_body if isinstance(request_body, str) else None,
                timeout=timeout
            )
        else:
            return f"Unsupported HTTP method: {method}"
        
        # Format the response 
        try:
            response_body = response.json()
            formatted_body = json.dumps(response_body, indent=2)
        except (json.JSONDecodeError, ValueError):
            formatted_body = response.text
        
        result = f"Status Code: {response.status_code}\n"
        result += f"Headers: {dict(response.headers)}\n"
        result += f"Body: {formatted_body}"
        
        return result
    
    except requests.RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        import traceback
        return f"Error executing REST API request: {str(e)}\n{traceback.format_exc()}"