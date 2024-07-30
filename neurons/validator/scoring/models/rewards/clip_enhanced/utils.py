import json
from typing import List, Dict, Union

from openai.types.shared_params import FunctionDefinition
from openai.types.chat.chat_completion_named_tool_choice_param import (
    ChatCompletionNamedToolChoiceParam,
    Function,
)
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)


from neurons.validator.config import get_openai_client


def get_prompt_breakdown_function():
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


async def break_down_prompt(
    prompt: str, service: str = "openai"
) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    query = [
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

    tool = get_prompt_breakdown_function()

    if service == "openai":
        response = await get_openai_client().chat.completions.create(
            model="gpt-4o",
            temperature=0,
            tool_choice=ChatCompletionNamedToolChoiceParam(
                type="function",
                function=Function(name=tool.function.name),
            ),
            tools=[tool],
            messages=query,
        )
        return json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )
    elif service == "corcel":
        # Implement Corcel API call here
        pass
    else:
        raise ValueError("Invalid service specified")
