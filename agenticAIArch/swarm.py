from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
import os
import sys 
from dotenv import load_dotenv
import asyncio

load_dotenv()


os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

class SwarmState(TypedDict):
    messages : Annotated[List[BaseMessage],add_messages]
    active_agent : str

@tool
def transfer_to_math() -> str:
    """Transfers control to the Math Agent to perform calculations"""
    return 'Control transferred to MathAgent. Please solve the mathematical query'

@tool
def transfer_to_weather() -> str:
    """Transfers control to the Weather Agent to get weather information"""
    return 'Control transferred to WeatherAgent. Please get the weather information'

@tool
def transfer_to_triage() -> str:
    """Transfers control to the Triage Agent to get weather information"""
    return 'Control transferred to TriageAgent. Please co-ordinate the next steps with the user.'



async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                'command': sys.executable,
                'args' : ['mathserver.py'],
                'transport': 'stdio'
            },
            'weather': {
                'url' : 'http://localhost:8000/mcp',
                'transport' : 'streamable-http'
            }
        }
    )
    tools = await client.get_tools()
    model = ChatGroq(model_name='llama-3.3-70b-versatile')
    # 1. Filter weather and math tools from MCP
    weather_tools = [t for t in tools if t.name == 'get_weather']
    math_tools = [t for t in tools if t.name in ['add', 'subtract', 'multiply', 'divide']]

    triage_swarm_tools = [transfer_to_math, transfer_to_weather]
    weather_swarm_tools = weather_tools + [transfer_to_math, transfer_to_triage]
    math_swarm_tools = math_tools + [transfer_to_weather, transfer_to_triage]


    async def triage_node(state: SwarmState):
        print('--- Triage Agent is running ---')
        
        prompt = f'''
        You are the Triage Agent, the entry point of the swarm.
        If user wantes the weather, call transfer_to_weather. 
        If user wants math calculations, call transfer_to_math.
        Else, answer normally.
        '''

        messages = [('system', prompt)] + state['messages']
        triage_agent = create_agent(model, triage_swarm_tools)
        response = await triage_agent.ainvoke({'messages': messages})
        new_messages = response['messages'][len(messages):]
        return {'messages': new_messages, 'active_agent': 'TriageAgent'}


    async def weather_node(state: SwarmState):
        print('---Weather Agent is running---')
        prompt = f'''
        You are a Weather Agen and, your job is to fetch weather using get_weather.
        If you need math skills call transfer_to_math.
        When done, call transfer_to_triage. 
        '''

        messages = [('system', prompt)] + state['messages']
        weather_agent = create_agent(model, weather_swarm_tools)
        response = await weather_agent.ainvoke({'messages': messages})
        
        new_messages = response['messages'][len(messages):]
        return {'messages': new_messages, 'active_agent': 'WeatherAgent'}


    async def math_node(state: SwarmState):
        print('---Math Agent is running---')
        prompt = f'''
        You are a Math Agent, your job is to perform math calculations using math tools.
        If you need weather information call transfer_to_weather.
        When done, call transfer_to_triage.
        '''
        messages = [('system', prompt)] + state['messages']
        math_agent = create_agent(model, math_swarm_tools)
        response = await math_agent.ainvoke({'messages': messages})
        new_messages = response['messages'][len(messages):]
        return {'messages': new_messages, 'active_agent': 'MathAgent'}

    
    all_swarm_tools = tools + [transfer_to_math, transfer_to_weather, transfer_to_triage]
    tool_node = ToolNode(all_swarm_tools)

    def route_after_agent(state:SwarmState) -> Literal['TriageAgent','WeatherAgent','MathAgent']:
        if state['messages'][-1].tool_calls:
            return 'tools'
        
        return END

    def route_after_tools(state: SwarmState) -> Literal["TriageAgent", "WeatherAgent", "MathAgent"]:
        # Find the last AIMessage to see what tool was requested
        for msg in reversed(state["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_name = msg.tool_calls[0]["name"]
                if tool_name == "transfer_to_math":
                    return "MathAgent"
                elif tool_name == "transfer_to_weather":
                    return "WeatherAgent"
                elif tool_name == "transfer_to_triage":
                    return "TriageAgent"
                break
        # If it was a standard tool, return control to the active agent
        return state["active_agent"]
    
        # 9. Initialize Graph
    workflow = StateGraph(SwarmState)

    # 10. Add all nodes
    workflow.add_node("TriageAgent", triage_node)
    workflow.add_node("WeatherAgent", weather_node)
    workflow.add_node("MathAgent", math_node)
    workflow.add_node("tools", tool_node)

    # 11. Define Swarm flow
    workflow.add_edge(START, "TriageAgent")

    # Conditional routing after each agent
    workflow.add_conditional_edges(
        "TriageAgent",
        route_after_agent,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    workflow.add_conditional_edges(
        "WeatherAgent",
        route_after_agent,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    workflow.add_conditional_edges(
        "MathAgent",
        route_after_agent,
        {
            "tools": "tools",
            "__end__": END
        }
    )

    # Dynamic routing after tool execution (directing to the correct active agent)
    workflow.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "TriageAgent": "TriageAgent",
            "WeatherAgent": "WeatherAgent",
            "MathAgent": "MathAgent"
        }
    )

    # 12. Compile
    app = workflow.compile()

    # 13. Invoke the Swarm
    initial_state = {
        "messages": [
            ("user", "Compare the weather in New York and Los Angeles, and tell me the absolute difference in temperature.")
        ],
        "active_agent": "TriageAgent"
    }

    print("\n--- Starting Swarm Workflow ---")
    result = await app.ainvoke(initial_state)

    print("\n=== FINAL CONVERSATION RESULT ===")
    for msg in result["messages"]:
        role = msg.type.upper() if hasattr(msg, 'type') else 'UNKNOWN'
        print(f"[{role}]: {msg.content}")


asyncio.run(main())

        

    
        