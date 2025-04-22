# agent_instagram.py
import os
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI

load_dotenv()

def create_agent():
    # Load DeepSeek API key from environment
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise ValueError("DEEPSEEK_API_KEY not set in your .env file")

    # Instantiate the ChatOpenAI client pointed at DeepSeek's API
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=deepseek_key
    )

    # Configure Chromium to use a persistent profile for consistent fingerprint & session
    profile_dir = os.path.abspath("ig_profile")
    config = BrowserConfig(
        headless=False,
        user_data_dir=profile_dir  # Playwright persistent context
    )
    browser = Browser(config=config)

    # Define the browsing task for the agent
    task_description = (
        "Navigate to https://www.instagram.com\n"
        "Log in using INSTAGRAM_USER and INSTAGRAM_PASS from the environment\n"
        "After successful login:\n"
        "  1. Check for any unread direct messages (DMs) and respond contextually.\n"
        "  2. Check notifications for new comments on recent posts and reply with context-aware responses."
    )

    # Create the agent, disable memory if not installed
    agent = Agent(
        task=task_description,
        llm=llm,
        browser=browser,
        enable_memory=False
    )
    return agent

async def main():
    # Ensure the profile directory exists
    os.makedirs(os.getenv("IG_PROFILE_DIR", "ig_profile"), exist_ok=True)
    agent = create_agent()
    try:
        await agent.run()
    except Exception as e:
        print(f"Agent execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())

# Usage:
# 1. In your .env file, set:
#    DEEPSEEK_API_KEY=your_deepseek_api_key
#    INSTAGRAM_USER=your_username
#    INSTAGRAM_PASS=your_password
# 2. Ensure the profile directory exists (it will be auto-created):
#    mkdir ig_profile
# 3. Install dependencies:
#    pip install -e .
#    pip install python-dotenv langchain-openai
# 4. Run the agent:
#    python agent_instagram.py
