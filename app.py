import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from PackageSafetyAnalyzerTool import PackageSafetyAnalyzer, PackageSafetyTool
from CodeParsingTool import CodeParsingTool
from langchain.schema import HumanMessage
import dotenv
import os

dotenv.load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o",
    temperature=0,
)

analyzer = PackageSafetyAnalyzer(github_token=GITHUB_TOKEN)
package_safety_tool = PackageSafetyTool(analyzer=analyzer)
code_parsing_tool = CodeParsingTool(llm=llm)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

st.set_page_config(page_title="LangChain Package Safety Agent", page_icon="🤖")
st.title("🤖 LangChain Package Safety Agent")
st.write("Enter text or Python code. The agent will analyze code imports for package safety or answer normally.")
status_code_detection = st.empty()
status_package_verification = st.empty()
status_llm_response = st.empty()

if "messages" not in st.session_state:
    st.session_state.messages = []

def render_chat_history():
    chat_str = ""
    for role, content in st.session_state.messages:
        if role == "user":
            chat_str += f"**You:**\n{content}\n\n"
        else:
            chat_str += f"**Agent:**\n{content}\n\n"
    return chat_str

chat_history = st.empty()
chat_history.markdown(render_chat_history())

with st.form(key="input_form", clear_on_submit=True):
    cols = st.columns([8, 1])
    user_input = cols[0].text_input("Your message", placeholder="Type your message or code here...")
    send_button = cols[1].form_submit_button("Send")

if send_button and user_input.strip():
    with st.spinner("Agent is processing..."):
        # Step 1: Detect code
        status_code_detection.info("Detecting if input contains code...")
        parse_result = code_parsing_tool._run(user_input)
        status_code_detection.success(f"Code detection result: {parse_result}")

        if not parse_result.get("is_code", False):
            status_llm_response.info("Generating response for non-code input...")
            answer = llm([HumanMessage(content=user_input)]).content
            status_llm_response.success("Response generated.")
            st.session_state.messages.append(("agent", answer))
        else:
            # Step 2: Verify packages
            language = parse_result.get("language")
            imports = parse_result.get("imports", [])
            status_package_verification.info(f"Verifying packages: {imports}")

            verification_results = analyzer.analyze_code(
                "\n".join(f"import {pkg}" for pkg in imports), lang=language
            )
            status_package_verification.success("Package verification completed.")

            verification_summary = "\n".join(
                f"- {pkg}: Valid={info.get('is_valid')}, Risk={info.get('risk')}"
                for pkg, info in verification_results.items()
            )

            # Step 3: Generate final answer
            combined_prompt = (
                f"User query:\n{user_input}\n\n"
                f"Detected language: {language}\n"
                f"Imported packages and verification:\n{verification_summary}\n\n"
                f"Please answer the user's query. Highlight any unverified or risky packages."
            )
            status_llm_response.info("Generating response with package info...")
            answer = llm([HumanMessage(content=combined_prompt)]).content
            status_llm_response.success("Response generated.")

            combined_response = f"{answer}\n\nPackage Verification:\n{verification_summary}"
            st.session_state.messages.append(("user", user_input))
            st.session_state.messages.append(("agent", combined_response))

            chat_history.markdown(render_chat_history())

elif send_button and not user_input.strip():
    st.warning("Please enter some input.")
