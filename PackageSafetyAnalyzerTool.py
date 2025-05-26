import requests
import ast
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.tools import BaseTool
import json
from pydantic import PrivateAttr

class PackageSafetyAnalyzer:
    name: str = "package_safety_checker"
    description: str = (
        "Use this tool to analyze Python code imports for package validity and risk. "
        "Input should be Python code as a string. Output is a JSON string with package analysis."
    )
    def __init__(self, config_path='config.json', github_token=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.known_packages = set(self.config.get("known_packages", []))
        self.PYPI_API = self.config.get("pypi_api", "https://pypi.org/pypi/{}/json")
        self.NPM_API = self.config.get("npm_api", "https://registry.npmjs.org/{}")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.github_token = github_token

    def detect_language(self, code):
        python_keywords = ['def ', 'import ', 'print(', 'class ', 'self']
        js_keywords = ['function ', 'console.log', 'var ', 'let ', 'const ']

        code_lower = code.lower()
        if any(k in code_lower for k in python_keywords):
            return 'python'
        elif any(k in code_lower for k in js_keywords):
            return 'javascript'
        else:
            return 'unknown'

    def validate_package(self, package_name, lang='python'):
        try:
            if lang == 'python':
                response = requests.get(self.PYPI_API.format(package_name))
            else:
                response = requests.get(self.NPM_API.format(package_name))
            return response.status_code == 200
        except Exception:
            return False

    def get_subpackages(self, package_name):
        try:
            response = requests.get(self.PYPI_API.format(package_name))
            if response.status_code != 200:
                return []
            data = response.json()
            top_level = data.get('info', {}).get('top_level', None)
            if top_level:
                if isinstance(top_level, list):
                    return top_level
                elif isinstance(top_level, str):
                    return [top_level]
            return []
        except Exception:
            return []

    def check_github_popularity(self, package_name):
        if not self.github_token:
            return None 

        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        try:
            url = f"https://api.github.com/search/repositories?q={package_name}+in:name&sort=stars&order=desc"
            r = requests.get(url, headers=headers)
            if r.status_code != 200:
                return None
            items = r.json().get('items', [])
            if not items:
                return None
            top_repo = items[0]
            return top_repo.get('stargazers_count', None)
        except Exception:
            return None

    def semantic_similarity(self, package_name):
        embeddings = self.embedder.encode([package_name] + list(self.known_packages))
        similarities = np.dot(embeddings[0], embeddings[1:].T)
        return float(max(similarities))

    def analyze_code(self, code, lang='python'):
        try:
            tree = ast.parse(code)
        except Exception as e:
            return {"error": f"Code parsing error: {e}"}

        packages = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    packages.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    packages.add(node.module.split('.')[0])

        results = {}
        for pkg in packages:
            exists = self.validate_package(pkg, lang)
            similarity = self.semantic_similarity(pkg)
            stars = self.check_github_popularity(pkg)
            risk = 'low'
            if not exists and similarity > 0.7:
                risk = 'high'
            elif not exists:
                risk = 'medium'
            elif stars is not None and stars < 50:
                risk = 'medium'

            results[pkg] = {
                "is_valid": int(exists),
                "similarity": similarity,
                "github_stars": stars,
                "risk": risk
            }

        return results


class PackageSafetyTool(BaseTool):
    name: str = "package_safety_checker"
    description: str = (
        "Use this tool to analyze Python code imports for package validity and risk. "
        "Input should be Python code as a string. Output is a JSON string with package analysis."
    )
    _analyzer: object = PrivateAttr()
    
    def __init__(self, analyzer: PackageSafetyAnalyzer):
        super().__init__()
        self._analyzer = analyzer

    def _run(self, query: str) -> str:
        results = self._analyzer.analyze_code(query, lang='python')
        return json.dumps(results, indent=2)

    async def _arun(self, query: str) -> str:
        return self._run(query)