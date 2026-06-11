from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import asyncio
import sys
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

class PipelineState(TypedDict):
    city_a:str
    city_b:str
    weather_data_a:str
    weather_data_b:str
    temp_diff:float
    final_report:str

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

    # Sequential Agents Wflow:

    weather_tools = [t for t in tools if t.name=='get_weather']
    math_tools = [t for t in tools if t.name in ['add','subtract','multiply','divide']]  



    model = ChatGroq(model_name='llama-3.3-70b-versatile')

    weather_agent = create_agent(model,weather_tools)
    math_agent = create_agent(model,math_tools)


    async def fetch_weather_node(state:PipelineState):
        print(f"Fetching weather data for {state['city_a']} and {state['city_b']} --")
        resp_a = await weather_agent.ainvoke({'messages':[{'role':'user', 'content':f'What is the weather in {state["city_a"]}? Try to extract the temperature numerical value if available.'}]})
        resp_b = await weather_agent.ainvoke({'messages':[{'role':'user', 'content':f'What is the weather in {state["city_b"]}? Try to extract the temperature numerical value if available.'}]})
        return {
            'weather_data_a': resp_a['messages'][-1].content,
            'weather_data_b': resp_b['messages'][-1].content
        }

    async def calculate_temp_diff_node(state:PipelineState):
        print(f"Calculating temperature difference between {state['city_a']} and {state['city_b']} --")

        userPrompt = f'''
        Here is the weather of 2 cities:\n
        City A: {state['weather_data_a']}\n
        City B: {state['weather_data_b']}\n\n
        Calculate the absolute difference in temperature between City A and B, using the subtract tool to perform
        the calculation. Ensure you output the final answer clearly.
        '''

        resp = await math_agent.ainvoke({'messages':[{'role':'user', 'content':userPrompt}]})

        final = resp['messages'][-1].content

        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", final)
        temp_diff = float(numbers[0]) if numbers else 0.0
        
        return {
            'temp_diff': temp_diff
        }

    async def generate_report_node(state:PipelineState):
        print(f"Generating final report --")

        userPrompt = f'''
        Generate a beautiful, summary report of the temperature difference between 2 cities.\n\n

        City A: {state['city_a']}
        Weather A: {state['weather_data_a']}\n
        City B: {state['city_b']}
        Weather B: {state['weather_data_b']}\n
        Temperature Difference: {state['temp_diff']}\n\n

        Write a friendly summary highlighting the weather in both places and confirming the temperature difference.

        ''' 

        resp = await model.ainvoke([("user", userPrompt)])
        return {'final_report': resp.content}

        #Intialize graph
    workflow = StateGraph(PipelineState)

    #Add nodes
    workflow.add_node('fetch_weather',fetch_weather_node)
    workflow.add_node('calculate_temp_diff',calculate_temp_diff_node)
    workflow.add_node('generate_report',generate_report_node)

    # Add edges
    workflow.add_edge(START,'fetch_weather')
    workflow.add_edge('fetch_weather','calculate_temp_diff')
    workflow.add_edge('calculate_temp_diff','generate_report')
    workflow.add_edge('generate_report',END)

    # now compile the graph
    app = workflow.compile()

    initial_state = {

        'city_a':'New York',
        'city_b':'Los Angeles'
    }

    print('--Staerting Sequential Workflow--')
    result = await app.ainvoke(initial_state)

    print(result['final_report'])

            



        

        
        
    


    # agent = create_agent(
    #     model,
    #     tools,
    #     system_prompt=(
    #         "You are a helpful assistant. You must NOT nest tool calls (i.e., do not pass "
    #         "a tool call as an argument to another tool). If a calculation requires multiple "
    #         "steps, perform them sequentially one at a time. For example, to calculate "
    #         "(83+500)*105, first call the add tool with 83 and 500, wait for the response, "
    #         "and then in the next turn call the multiply tool with that result and 105."
    #     )
    # )

    # mathResp = await agent.ainvoke({'messages':[{'role':'user','content':'What is (83+500)x105?'}]})

    # weatherResp = await agent.ainvoke({'messages':[{'role':'user','content':'What is the weather like in NYC? '}]})

    # print('Math Question response: ', mathResp['messages'][-1].content)
    
    # print('Weather question response: ', weatherResp['messages'][-1].content)



asyncio.run(main())
    

    