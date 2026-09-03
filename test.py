# from cryptography.fernet import Fernet


# def encrypt(raw_key: str) -> str:
#     # ponytail: plug real encryption here (e.g. Fernet.encrypt) when a KMS
#     # key/secret is available. Keeping this isolated per design rule #3.
#     master_key = "J8ZvAsmsSCiEg_U25hylZQo7IVTyyAzGfXZY-DjQQ0U="
#     fernet = Fernet(master_key)

#     encrypted_key = fernet.encrypt(raw_key.encode())
#     return encrypted_key.decode()


# def decrypt(stored_value: str) -> str:
#     # ponytail: plug real decryption here to match encrypt() above.

#     master_key = "J8ZvAsmsSCiEg_U25hylZQo7IVTyyAzGfXZY-DjQQ0U="
#     fernet = Fernet(master_key)

#     return fernet.decrypt(stored_value).decode()


# encrypt("sk-KEzLkDC9IkYiDlRRr4KjX0tvoaUsQDNw2gg0b88PgUJTVemSFGGNSOpc9ABZWNqO")
# decrypt(
#     "gAAAAABqhs0W2NIVGpRpOWKdPO-dEI-653zqo5NTRYAaNl4KrLAvgDIX7MypMcZfWJEyhPSdWiruHqx2WMlnAnkLIo7yZskSL3RIdDMthYs2iH5bB95FpbtWaA70tE_PJtdp_kE4ylux1bAllcWrSWZ6eeeRHitEcT69GwK1rw9mgNMk_wrQ_Nw="
# )


import asyncio

from pydantic_ai import Agent, ModelRequestNode
from pydantic_ai.agent.abstract import Instructions
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pathlib import Path
import yaml
from pydantic_graph import End

def capital(text:str):
    """Use when asked about capitals"""
    return "capital of france is paris"

provider = OpenAIProvider(
    base_url="https://opencode.ai/zen/v1",
    api_key="sk-KEzLkDC9IkYiDlRRr4KjX0tvoaUsQDNw2gg0b88PgUJTVemSFGGNSOpc9ABZWNqO",
)
model = OpenAIChatModel("nemotron-3-ultra-free", provider=provider)



agent = Agent(model, tools=[capital], system_prompt="You are a helpful assistant.")

async def main():
  async with agent.iter('What is the capital of France?') as agent_run:
    #   node = agent_run.next_node  
    #   all_nodes = [node]
      # Drive the iteration manually:
      async for node in agent_run:
        if isinstance(node, ModelRequestNode):
          async with node.stream(agent_run.ctx) as stream:
                  async for chunk in stream:
                      print(chunk, end="")
    #   all_nodes = [node]
    #   while not isinstance(node, End):  
    #       node = await agent_run.next(node)  
    #       all_nodes.append(node)  
    #   print(all_nodes)
      
asyncio.run(main())


# [UserPromptNode(user_prompt='What is the capital of France?', instructions_functions=[], system_prompts=('You are a helpful assistant.',), system_prompt_functions=[], system_prompt_dynamic_functions={}),
#  ModelRequestNode(request=ModelRequest(parts=[SystemPromptPart(content='You are a helpful assistant.', timestamp=datetime.datetime(2026, 9, 1, 11, 2, 20, 284529, tzinfo=datetime.timezone.utc)),
#                                               UserPromptPart(content='What is the capital of France?', timestamp=datetime.datetime(2026, 9, 1, 11, 2, 20, 284538, tzinfo=datetime.timezone.utc))], timestamp=datetime.datetime(2026, 9, 1, 11, 2, 20, 284981, tzinfo=datetime.timezone.utc),run_id='01a05ca2-5f73-75e8-90bf-ac9d848d21a0', conversation_id='01a05ca2-5f73-75e8-90bf-ac9edf3f7236')),
#  CallToolsNode(model_response=ModelResponse(parts=[ThinkingPart(content='The user is asking about the capital of France. I should use the capital function to answer this question.', id='reasoning_content', provider_name='openai'),
#                                                    ToolCallPart(tool_name='capital', args='{"text": "France"}', tool_call_id='call_832085eeae4f46d0a43df734')],
#                                             usage=RequestUsage(details={'reasoning_tokens': 0}, input_tokens=265, output_tokens=43), model_name='mimo-v2.5-free', timestamp=datetime.datetime(2026, 9, 1, 11, 2, 27, 491386, tzinfo=datetime.timezone.utc), provider_name='openai', provider_url='https://opencode.ai/zen/v1/', provider_details={'finish_reason': 'tool_calls', 'timestamp': datetime.datetime(2026, 9, 1, 11, 2, 27, tzinfo=TzInfo(0))}, provider_response_id='8178dd28-ca55-4133-916f-8e48206d0578_dfb224ccc204459382aab0ab9594232f', finish_reason='tool_call', run_id='01a05ca2-5f73-75e8-90bf-ac9d848d21a0', conversation_id='01a05ca2-5f73-75e8-90bf-ac9edf3f7236')),
#  ModelRequestNode(request=ModelRequest(parts=[ToolReturnPart(tool_name='capital', content='capital of france is paris', tool_call_id='call_832085eeae4f46d0a43df734', timestamp=datetime.datetime(2026, 9, 1, 11, 2, 27, 515158, tzinfo=datetime.timezone.utc))], timestamp=datetime.datetime(2026, 9, 1, 11, 2, 27, 515650, tzinfo=datetime.timezone.utc), run_id='01a05ca2-5f73-75e8-90bf-ac9d848d21a0', conversation_id='01a05ca2-5f73-75e8-90bf-ac9edf3f7236')),
#  CallToolsNode(model_response=ModelResponse(parts=[ThinkingPart(content="The function returned that the capital of France is Paris. I'll provide this information to the user.", id='reasoning_content', provider_name='openai'), TextPart(content='The capital of France is **Paris**.')], usage=RequestUsage(details={'reasoning_tokens': 0}, input_tokens=325, output_tokens=32), model_name='mimo-v2.5-free', timestamp=datetime.datetime(2026, 9, 1, 11, 2, 29, 431963, tzinfo=datetime.timezone.utc), provider_name='openai', provider_url='https://opencode.ai/zen/v1/', provider_details={'finish_reason': 'stop', 'timestamp': datetime.datetime(2026, 9, 1, 11, 2, 29, tzinfo=TzInfo(0))}, provider_response_id='c3457668-63e9-4e83-bc72-81d9b46cde8b_d6949fac23c543a282d4021824c72289', finish_reason='stop', run_id='01a05ca2-5f73-75e8-90bf-ac9d848d21a0', conversation_id='01a05ca2-5f73-75e8-90bf-ac9edf3f7236')),
#  End(data=FinalResult(output='The capital of France is **Paris**.'))]