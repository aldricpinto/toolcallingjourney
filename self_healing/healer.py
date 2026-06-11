import os
import re
import argparse
import subprocess
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load API keys and config from environment
load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

# Regex patterns for different language stack traces
TRACEBACK_PATTERNS = [
    # Python: File "path/to/file.py", line 12, in <module>
    re.compile(r'File "(?P<file>[^"]+\.py)", line (?P<line>\d+)'),
    
    # Node.js: at Object.<anonymous> (/path/to/file.js:12:34) or /path/to/file.js:12
    re.compile(r'(?P<file>\/[^\s:]+|[a-zA-Z]:\\[^\s:]+\.[a-zA-Z0-9]+):(?P<line>\d+)'),
    
    # Go: /path/to/file.go:12 +0x64
    re.compile(r'(?P<file>\/[^\s:]+|[a-zA-Z]:\\[^\s:]+\.go):(?P<line>\d+)'),
]

# Pydantic schema for structured output from OpenRouter
class FilePatch(BaseModel):
    file_path: str = Field(description="The absolute path of the file to modify.")
    new_content: str = Field(description="The full, updated code contents of the file with the bug fixed. Do NOT truncate or leave placeholders.")
    explanation: str = Field(description="Brief explanation of the error and how this patch fixes it.")

class HealerPatchResponse(BaseModel):
    patches: List[FilePatch] = Field(description="List of file patches generated to resolve the error.")

class HealerRunner:
    def __init__(self, command: str, watch: bool = False, max_attempts: Optional[int] = None, git_mode: str = "commit"):
        self.command = command
        self.watch = watch
        self.max_attempts = max_attempts if max_attempts is not None else MAX_ATTEMPTS
        self.git_mode = git_mode  # "none" | "stage" | "commit"
        self.workspace_dir = os.getcwd()
        
        # Initialize LLM client from global environment variables
        if not API_KEY:
            print("[Warning] No API key (OPENROUTER_API_KEY or OPENAI_API_KEY) found. LLM calls will fail.")
            
        self.llm = ChatOpenAI(
            base_url=LLM_BASE_URL,
            api_key=API_KEY,
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
        )
        self.structured_llm = self.llm.with_structured_output(HealerPatchResponse)

    def resolve_file_path(self, raw_path: str) -> Optional[str]:
        """Resolves a traceback path to an actual file in the workspace, skipping standard libraries."""
        abs_path = os.path.abspath(raw_path) if os.path.isabs(raw_path) else os.path.abspath(os.path.join(self.workspace_dir, raw_path))
        
        if os.path.exists(abs_path) and abs_path.startswith(self.workspace_dir):
            ignored_substrings = [".venv", "venv", "node_modules", "lib/python", "Cellar", "node:internal"]
            if not any(sub in abs_path for sub in ignored_substrings):
                return abs_path
        return None

    def extract_failing_files(self, stderr: str) -> Dict[str, List[int]]:
        """Parses the stderr log to find workspace source files and their respective line numbers."""
        failing_files = {}
        
        for pattern in TRACEBACK_PATTERNS:
            for match in pattern.finditer(stderr):
                raw_path = match.group("file")
                line_no = int(match.group("line"))
                
                resolved = self.resolve_file_path(raw_path)
                if resolved:
                    if resolved not in failing_files:
                        failing_files[resolved] = []
                    failing_files[resolved].append(line_no)
                    
        return failing_files

    async def invoke_agent_repair(self, traceback: str, failing_files: Dict[str, List[int]]) -> bool:
        """Calls OpenRouter LLM to generate code repairs and applies them to files."""
        print(f"[Healer] Contacting self-healing agent on OpenRouter ({LLM_MODEL}) ...")
        
        # Read the contents of all files involved in the traceback
        code_context = ""
        for path, lines in failing_files.items():
            rel_path = os.path.relpath(path, self.workspace_dir)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                code_context += f"--- FILE: {rel_path} (Absolute Path: {path}) ---\n"
                code_context += f"Failing lines reference: {lines}\n"
                code_context += "```python\n" if path.endswith(".py") else "```javascript\n"
                code_context += content + "\n```\n\n"
            except Exception as e:
                print(f"[Healer] Error reading file {rel_path} for LLM context: {e}")
                return False

        system_prompt = (
            "You are an expert software developer and self-healing runtime assistant.\n"
            "Your task is to analyze a program crash (traceback/logs) and the source code of the affected files, "
            "identify the bug, and output the fully corrected file content to fix the issue.\n"
            "Follow these critical guidelines:\n"
            "1. You must return the COMPLETE code of the file. DO NOT truncate, use placeholders, or skip lines.\n"
            "2. Ensure all imports and helper functions are preserved.\n"
            "3. Only modify the code necessary to fix the error in the traceback.\n"
            "4. Retain all comments that are unrelated to the code changes."
        )

        user_prompt = (
            f"The application crashed while running this command:\n"
            f"```bash\n{self.command}\n```\n\n"
            f"Here is the traceback / error logs:\n"
            f"```\n{traceback}\n```\n\n"
            f"Here is the source code of the files involved in the stack trace:\n\n"
            f"{code_context}\n"
            f"Please fix the error and return the patched file contents using the requested schema."
        )

        try:
            # Call the LLM to get structured JSON repair patches
            response = await self.structured_llm.ainvoke([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            
            if not response or not response.patches:
                print("[Healer] Agent returned no patches.")
                return False
                
            for patch in response.patches:
                resolved_path = self.resolve_file_path(patch.file_path)
                if not resolved_path:
                    print(f"[Healer] Warning: LLM patch tried to edit non-workspace file: {patch.file_path}")
                    continue
                    
                rel_path = os.path.relpath(resolved_path, self.workspace_dir)
                print(f"[Healer] Applying patch to {rel_path}...")
                print(f"[Healer] Agent explanation: {patch.explanation}")
                
                # Overwrite file with corrected content
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(patch.new_content)
                    
            return True
            
        except Exception as e:
            print(f"[Healer] LLM invocation failed: {e}")
            return False

    async def run_loop(self) -> bool:
        """Executes the target command in a loop up to max_attempts, repairing crashes iteratively."""
        attempt = 1
        all_patched_files: List[str] = []
        last_traceback: str = ""

        while attempt <= self.max_attempts:
            print(f"\n[Healer] Execution Attempt {attempt}/{self.max_attempts}")
            print(f"[Healer] Running command: {self.command}")

            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True
            )

            print(f"[Healer] Exit code: {result.returncode}")

            if result.returncode == 0:
                print("[Healer] SUCCESS! Command ran successfully without errors.")
                if all_patched_files:
                    self.git_action(all_patched_files, last_traceback)
                return True

            combined_logs = result.stderr + "\n" + result.stdout
            last_traceback = combined_logs
            failing_files = self.extract_failing_files(combined_logs)

            if not failing_files:
                print("[Healer] FAILURE: Command failed, but no workspace files were identified in stack trace logs.")
                print(f"[Healer] Stderr preview:\n{result.stderr[:400]}")
                return False

            # Log failing files
            print("[Healer] Detected failure in:")
            for path, lines in failing_files.items():
                print(f"  - {os.path.relpath(path, self.workspace_dir)} at line(s) {lines}")

            # Attempt to repair the code
            repaired = await self.invoke_agent_repair(combined_logs, failing_files)
            if not repaired:
                print("[Healer] Repair failed or was aborted.")
                return False

            # Track all files patched across iterations for the final git commit
            all_patched_files.extend([p for p in failing_files.keys() if p not in all_patched_files])
            attempt += 1

        print(f"[Healer] Reached maximum repair attempts ({self.max_attempts}) without success.")
        return False

    def git_action(self, patched_files: List[str], traceback_summary: str):
        """Handles git staging/committing based on the selected git_mode."""
        if self.git_mode == "none":
            print("[Healer][Git] Mode: none — skipping all git operations.")
            return

        try:
            # Stage patched files (both 'stage' and 'commit' modes)
            staged = []
            for path in patched_files:
                rel = os.path.relpath(path, self.workspace_dir)
                result = subprocess.run(["git", "add", rel], cwd=self.workspace_dir, capture_output=True, text=True)
                if result.returncode == 0:
                    staged.append(rel)
                    print(f"[Healer][Git] Staged: {rel}")
                else:
                    print(f"[Healer][Git] Warning: Could not stage {rel}: {result.stderr.strip()}")

            if self.git_mode == "stage":
                print(f"[Healer][Git] Mode: stage — {len(staged)} file(s) staged. Run 'git commit' when ready.")
                return

            # Full commit (mode == "commit")
            error_line = next(
                (l.strip() for l in traceback_summary.splitlines() if l.strip() and not l.startswith(" ")),
                "Unknown error"
            )[:120]
            commit_msg = f"[AI Self-Heal] Fixed: {error_line}"

            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.workspace_dir,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"[Healer][Git] Committed: '{commit_msg}'")
            else:
                print(f"[Healer][Git] Commit failed: {result.stderr.strip()}")

        except FileNotFoundError:
            print("[Healer][Git] Git not found on this system. Skipping git operations.")
        except Exception as e:
            print(f"[Healer][Git] Unexpected error: {e}")

    def start(self):
        """Main runner entrypoint"""
        import asyncio
        if self.watch:
            print("[Healer] Watch mode is not implemented yet. Running in standard loop mode.")
        asyncio.run(self.run_loop())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic Self-Healing CLI Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("cmd", help="The terminal command to run (e.g. 'python pipeline.py')")
    parser.add_argument("--watch", action="store_true", help="Monitor dev servers in real-time")
    parser.add_argument("--attempts", type=int, default=None, help="Max repair loops (overrides MAX_ATTEMPTS in .env)")
    parser.add_argument(
        "--git",
        choices=["none", "stage", "commit"],
        default="commit",
        metavar="MODE",
        help=(
            "Git behaviour after a successful repair:\n"
            "  none   — don't touch git at all\n"
            "  stage  — git add only (you review and commit yourself)\n"
            "  commit — git add + git commit (default, fully automated)"
        )
    )

    args = parser.parse_args()

    runner = HealerRunner(
        command=args.cmd,
        watch=args.watch,
        max_attempts=args.attempts,
        git_mode=args.git
    )
    runner.start()
