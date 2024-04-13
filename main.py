import warnings

# Ignore DeprecationWarning


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,SystemMessage

from dotenv import load_dotenv

import os

load_dotenv()

MODEL_NAME = "gemini-1.5-pro-latest"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

# result = llm.invoke("Write a ballad about LangChain")
# print(result.content)


def multimode_chat(text="", image_url=""):
    
    if text and not image_url:
        message = HumanMessage(content=text)
    elif image_url and not text:
        message = HumanMessage(content=[{"type": "image_url", "image_url": image_url}])
    else:
        message = HumanMessage(content=[{"type": "text", "text": text}, {"type": "image_url", "image_url": image_url}])
    
    result = llm.invoke([message])
    
    return result.content

print(multimode_chat(image_url="image.jpg",text = "Based on this image write a story about in 500 words"))

def gemini_agent(text ="",system_prompt=""):
    warnings.filterwarnings("ignore")
    model = ChatGoogleGenerativeAI(model=MODEL_NAME)
    model(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ]
    )
    result = model.invoke(text)
    return result.content



# system_prompt = "Answer only 0 or 1. 0 means no, 1 means yes."

# text = "Is apple a fruit?"
# print(gemini_agent(text=text,system_prompt=system_prompt))

# with open("SHERLOCK_HOLMES.txt", "r") as file:
#     sherlock_holmes_story = file.read().strip()


# system_prompt = "You are a story summerizer who summrise the story in the following text. Answer only the summary in 200 words."

# print(gemini_agent(text=sherlock_holmes_story,system_prompt=system_prompt))
