import requests
from requests.exceptions import RequestException

"""_summary_
You are a weather assistant that provides current weather data for user-specified locations using only the Weather_Tool, which expects latitude and longitude. Infer the coordinates from the location yourself. If the user provides coordinates, infer the approximate location and refer to it in your response.
"""

def tool_config():
    return {
        "toolSpec": {
            "name": "weather_tool",
            "description": "Get the current weather for a given location, based on its WGS84 coordinates.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "string",
                            "description": "Geographical WGS84 latitude of the location.",
                        },
                        "longitude": {
                            "type": "string",
                            "description": "Geographical WGS84 longitude of the location.",
                        },
                    },
                    "required": ["latitude", "longitude"],
                }
            },
        }
    }


def invoke(input_data):
    """
    Fetches weather data for the given latitude and longitude using the Open-Meteo API.
    Returns the weather data or an error message if the request fails.

    :param input_data: The input data containing the latitude and longitude.
    :return: The weather data or an error message.
    """
    endpoint = "https://api.open-meteo.com/v1/forecast"
    latitude = input_data.get("latitude")
    longitude = input_data.get("longitude", "")
    params = {"latitude": latitude, "longitude": longitude, "current_weather": True}

    try:
        response = requests.get(endpoint, params=params)
        weather_data = {"weather_data": response.json()}
        response.raise_for_status()
        return {"json":weather_data}
    except RequestException as e:
        if e.response is not None:
            return e.response.json()
        else:
            return {"json":{"error": "Request failed", "message": str(e)}}
    except Exception as e:
        return {"json":{"error": type(e), "message": str(e)}}

