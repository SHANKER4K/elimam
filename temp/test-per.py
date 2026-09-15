from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.openai import OpenAIProvider


class Answer(BaseModel):
    text: str = Field(description="الإجابة النصية")
    confidence: float = Field(description="درجة الثقة بين 0 و1")


class Refusal(BaseModel):
    reason: str = Field(description="سبب الرفض أو الاعتذار")


provider = DeepSeekProvider(
    api_key="sk-aa1af21fba8146cc9f689caeb906b532",
)
model = OpenAIChatModel("deepseek-ai/DeepSeek-V4-Flash-0731", provider=provider)

agent = Agent(
    model,
    output_type=NativeOutput(
        Answer,
        name="answer_output",
        description="إرجاع إجابة مهيكلة تحتوي على نص ودرجة ثقة",
    ),
)

result = agent.run_sync("ما عاصمة الجزائر؟ وما درجة ثقتك؟")
answer: Answer = result.output
print(answer.text, answer.confidence)
