from mcp.server.fastmcp import FastMCP
import random

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get the weather for a city
    """
    return f"The weather in {city} is {random.choice(['sunny', 'cloudy', 'rainy', 'windy'])} and the temperature is {random.randint(10, 30)} degrees Celsius."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")