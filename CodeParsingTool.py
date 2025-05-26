from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import json

class CodeParsingTool(BaseTool):
    name: str = "code_parsing_tool"
    description: str = "Detects if input contains code, identifies language and imports, returns JSON."
    llm: ChatOpenAI = None
    
    def __init__(self, llm: ChatOpenAI):
        super().__init__()
        self.llm = llm

    def _run(self, query: str) -> dict:
        prompt = f"""
You receive an input string. Determine if it contains code or programming-related content.

If it contains code, identify the programming language (python, javascript, etc.) and list all imported packages or modules.

Return a JSON object with the following format:

{{
"is_code": true or false,
"language": "python" or "javascript" or null,
"imports": ["package1", "package2", ...]
}}

If no code is detected, return:

{{
"is_code": false,
"language": null,
"imports": []
}}

Input:
\"\"\"{query}\"\"\"
        """
        text_response = self.llm([HumanMessage(content=prompt)])
        text_response = text_response.content.strip()
        try:
            json_start = text_response.find("{")
            json_end = text_response.rfind("}") + 1
            json_str = text_response[json_start:json_end]
            result = dict(json.loads(json_str))
            return result
        except Exception as e:
            return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": text_response}

    async def _arun(self, query: str) -> dict:
        return self._run(query)