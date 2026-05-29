from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import asyncio
import sys
load_dotenv()


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

    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

    tools = await client.get_tools()

    model = ChatGroq(model_name='llama-3.3-70b-versatile')

    agent = create_agent(
        model,
        tools,
        system_prompt=(
            "You are a helpful assistant. You must NOT nest tool calls (i.e., do not pass "
            "a tool call as an argument to another tool). If a calculation requires multiple "
            "steps, perform them sequentially one at a time. For example, to calculate "
            "(83+500)*105, first call the add tool with 83 and 500, wait for the response, "
            "and then in the next turn call the multiply tool with that result and 105."
        )
    )

    mathResp = await agent.ainvoke({'messages':[{'role':'user','content':'What is (83+500)x105?'}]})

    weatherResp = await agent.ainvoke({'messages':[{'role':'user','content':'What is the weather like in NYC? '}]})

    print('Math Question response: ', mathResp['messages'][-1].content)
    
    print('Weather question response: ', weatherResp['messages'][-1].content)



asyncio.run(main())
    

    