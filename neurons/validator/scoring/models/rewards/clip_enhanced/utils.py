from typing import List, Dict, Union, TypedDict, Callable, Awaitable
import json
import aiohttp
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
)
from openai.types.shared_params import FunctionDefinition


# Type definitions
class ElementDict(TypedDict):
    description: str
    importance: float


class PromptBreakdown(TypedDict):
    elements: List[ElementDict]


MessageDict = Dict[str, str]
BreakdownFunction = Callable[[str], Awaitable[PromptBreakdown]]

# Configuration functions (to be implemented in config.py)
from neurons.validator.config import (
    get_openai_client,
    get_corcel_api_key,
)


def get_prompt_breakdown_function() -> ChatCompletionToolParam:
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name="break_down_prompt",
            description="Break down an image prompt into key elements for CLIP",
            parameters={
                "type": "object",
                "properties": {
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {
                                    "type": "string",
                                    "description": "A key element in the image",
                                },
                                "importance": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "description": "Importance of this element",
                                },
                            },
                            "required": ["description", "importance"],
                        },
                        "description": "Key elements from the prompt",
                    }
                },
                "required": ["elements"],
            },
        ),
    )


def get_query_messages(
    prompt: str,
) -> List[
    Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]
]:
    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "Break down image prompts into key elements. "
                "Each element should be concise and evaluatable. "
                "Assign importance based on significance to the image."
            ),
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Break down this image prompt: {prompt}",
        ),
    ]


async def openai_breakdown(prompt: str) -> PromptBreakdown:
    client: AsyncOpenAI = get_openai_client()
    messages = get_query_messages(prompt)
    tool = get_prompt_breakdown_function()

    response = await client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        tool_choice={
            "type": "function",
            "function": {"name": tool.function.name},
        },
        tools=[tool],
        messages=messages,
    )

    return json.loads(
        response.choices[0].message.tool_calls[0].function.arguments
    )


async def corcel_breakdown(prompt: str) -> PromptBreakdown:
    api_key = get_corcel_api_key()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    messages = get_query_messages(prompt)
    tool = get_prompt_breakdown_function()

    payload = {
        "model": "corcel/text-davinci-003",
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": 0,
        "tools": [tool.dict()],
        "tool_choice": {
            "type": "function",
            "function": {"name": tool.function.name},
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.corcel.io/cortext/text",
            headers=headers,
            json=payload,
        ) as response:
            if response.status == 200:
                result = await response.json()
                return json.loads(
                    result["choices"][0]["message"]["tool_calls"][0][
                        "function"
                    ]["arguments"]
                )
            else:
                raise Exception(
                    f"Corcel API request failed with status {response.status}"
                )


async def break_down_prompt(
    prompt: str, service: str = "openai"
) -> PromptBreakdown:
    services: Dict[str, BreakdownFunction] = {
        "openai": openai_breakdown,
        "corcel": corcel_breakdown,
    }

    if service not in services:
        raise ValueError(f"Invalid service specified: {service}")

    return await services[service](prompt)
