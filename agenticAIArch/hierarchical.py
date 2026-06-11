from typing import List, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages 
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

class HierarchicalState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


class RouteResponse(BaseModel):
    next: Literal["MathAgent", "WeatherAgent", "FINISH"] = Field(
        description="Select the next specialist to run. Choose FINISH when you have the final answer."
    )
    

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

    supervisor_model = model.with_structured_output(RouteResponse)

    async def supervisor_node(state: HierarchicalState):
        print(f"--- Supervisor is deciding ---")
        
        system_prompt = (
            "You are a supervisor managing a conversation between two workers: 'MathAgent' and 'WeatherAgent'. "
            "Your job is to read the conversation history and decide who should act next. "
            "If you need weather info (like current temperature), route to 'WeatherAgent'. "
            "If you need to perform calculations or comparisons on the numbers, route to 'MathAgent'. "
            "If the user's question has been fully answered, route to 'FINISH'."
        )
        
        # Invoke the supervisor model with the prompt and the current history
        messages = [("system", system_prompt)] + state["messages"]
        decision = await supervisor_model.ainvoke(messages)
        
        print(f"Supervisor decided: {decision.next}")
        return {"next": decision.next}
    

    # 1. Filter tools for each worker
    weather_tools = [t for t in tools if t.name == 'get_weather']
    math_tools = [t for t in tools if t.name in ['add', 'subtract', 'multiply', 'divide']]
    # 2. Instantiate the specialized ReAct agents
    weather_agent = create_agent(model, weather_tools)
    math_agent = create_agent(model, math_tools)
    # 3. Define the Math Agent Node
    async def math_node(state: HierarchicalState):
        print("--- Math Agent is running ---")
        
        # 1. Grab the last message (which is the weather report)
        weather_info = state["messages"][-1].content
        
        # 2. Build a clean, isolated user request for the Math Agent
        prompt = (
            f"Here is the weather information:\n{weather_info}\n\n"
            f"Calculate the absolute difference in temperature between the two cities. "
            f"First extract the temperature values, then use your math tools to subtract one from the other. "
            f"Show the subtraction using your tool."
        )
        
        # 3. Invoke the math agent with this isolated prompt
        response = await math_agent.ainvoke({"messages": [("user", prompt)]})
        
        # 4. Return all messages from this agent's turn to append to history
        return {"messages": response["messages"]}
    # 4. Define the Weather Agent Node
    async def weather_node(state: HierarchicalState):
        print("--- Weather Agent is running ---")
        # Run the agent on the entire conversation history
        response = await weather_agent.ainvoke({"messages": state["messages"]})
        new_messages = response["messages"][len(state["messages"]):]
        return {"messages": new_messages}

    # 1. Initialize Graph
    workflow = StateGraph(HierarchicalState)
    # 2. Add supervisor and worker nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("MathAgent", math_node)
    workflow.add_node("WeatherAgent", weather_node)
    # 3. Router function for conditional edges
    def route_options(state: HierarchicalState) -> Literal["MathAgent", "WeatherAgent", "__end__"]:
        if state["next"] == "FINISH":
            return END
        return state["next"]
    # 4. Connect START to supervisor
    workflow.add_edge(START, "supervisor")
    # 5. Connect supervisor to the correct node based on its decision
    workflow.add_conditional_edges(
        "supervisor",
        route_options,
        {
            "MathAgent": "MathAgent",
            "WeatherAgent": "WeatherAgent",
            "__end__": END
        }
    )
    # 6. Connect workers back to supervisor to check what to do next
    workflow.add_edge("MathAgent", "supervisor")
    workflow.add_edge("WeatherAgent", "supervisor")
    # 7. Compile the graph
    app = workflow.compile()
    # 8. Run the hierarchical graph with a query that requires BOTH agents
    initial_state = {
        "messages": [
            ("user", "Compare the weather in New York and Los Angeles, and tell me the absolute difference in temperature.")
        ]
    }
    print("\n--- Starting Hierarchical Workflow ---")
    result = await app.ainvoke(initial_state)
    print("\n=== FINAL CONVERSATION RESULT ===")
    for msg in result["messages"]:
        # print the role and the content of each message in the thread
        role = msg.type.upper() if hasattr(msg, 'type') else 'UNKNOWN'
        print(f"[{role}]: {msg.content}")
    


asyncio.run(main())