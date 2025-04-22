# agent_instagram.py
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_env_var(name, prompt_text=None):
    """Retrieve an environment variable or prompt the user."""
    value = os.getenv(name)
    if not value and prompt_text:
        value = input(f"{prompt_text}: ")
    if not value:
        raise ValueError(f"Environment variable '{name}' not set and no fallback provided.")
    return value


def init_llm():
    """Initialize the LLM based on provider and model environment vars or prompt."""
    provider = os.getenv('LLM_PROVIDER', '').lower() or input('Select LLM provider (openai, deepseek, gemini): ').strip().lower()
    model = os.getenv('LLM_MODEL', '') or input(f"Enter model name for '{provider}': ")

    if provider == 'openai':
        key = get_env_var('OPENAI_API_KEY', 'Enter your OpenAI API key')
        return ChatOpenAI(model=model, api_key=key)
    if provider == 'deepseek':
        key = get_env_var('DEEPSEEK_API_KEY', 'Enter your DeepSeek API key')
        return ChatOpenAI(model=model, base_url='https://api.deepseek.com/v1', api_key=key)
    if provider == 'gemini':
        key = get_env_var('GEMINI_API_KEY', 'Enter your Gemini API key')
        return ChatGoogleGenerativeAI(model=model, api_key=key)
    raise ValueError(f"Unknown provider '{provider}'")

async def main():
    # Initialize LLM and Instagram credentials
    llm = init_llm()
    insta_user = get_env_var('INSTAGRAM_USER', 'Enter your Instagram username')
    insta_pass = get_env_var('INSTAGRAM_PASS', 'Enter your Instagram password')

    # Set up persistent profile directory
    profile_dir = Path('ig_profile').resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using profile directory: {profile_dir}")

    # Launch BrowserUse Browser with persistent context
    config = BrowserConfig(
        headless=False,
        user_data_dir=str(profile_dir)
    )
    browser = Browser(config=config)
    print("Browser launched with persistent profile.")

    # Define the browsing task
    task = (
        f"1. Go to https://www.instagram.com and accept cookies if prompted. "
        f"2. On the login form, enter username '{insta_user}' and password '{insta_pass}', then click Log In. "
        f"3. After Login,Open Notifications, for each new comment open reply field and reply thoughtfully until no new comments remain."
        f"3. Once Comments are clear, open DMs and check for new messages. If there are new messages receieved, reply to them and send. If the last message is from yourself, dont do anything. "
        f"5. Respond with no more than 5 words. Make it sound like a HUMAN.  "
        f"6. If you encounter a CAPTCHA, report it and stop the agent."
        f"7. You can be flirty with BANTER." 
        f"8. If you encounter a CAPTCHA, report it and stop the agent."
        f"9. Do not open any links or images sent."
    
         )

    # Create and run the agent
    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        enable_memory=False
    )
    try:
        result = await agent.run()
        print(f"Agent result: {result}")
    except KeyboardInterrupt:
        print("🛑 Interrupted: closing browser and saving session...")
    finally:
        # Always close the browser to flush state
        await browser.close()
        print(f"Session data saved in {profile_dir}")

    # Verify persistence
    cookies_file = profile_dir / 'Default' / 'Cookies'
    if cookies_file.exists():
        print(f"✅ Cookies file found: {cookies_file}")
    else:
        print("⚠️ No cookies file; persistence may have failed.")
    print("ig_profile contents:", [p.name for p in profile_dir.iterdir()])

if __name__ == '__main__':
    asyncio.run(main())

# Usage:
# 1. Set LLM_PROVIDER, LLM_MODEL, and their API_KEY env vars to skip prompts.
# 2. Set INSTAGRAM_USER and INSTAGRAM_PASS in .env or input when prompted.
# 3. Run: python agent_instagram.py
# The 'ig_profile' folder will now persist cookies/storage across runs.
