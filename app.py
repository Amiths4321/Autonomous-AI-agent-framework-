#!/usr/bin/env python3
"""
Autonomous AI Agent Framework (Ollama Remoted Variant)
Chains an external Ollama GPU instance with native tool calls to accomplish goals.

Usage:
    python remote_ollama_agent.py "Find the area in sqft of the largest room in a house."
    python remote_ollama_agent.py   (interactive mode)
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from ollama import Client 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(
    title="Multimodal Architecture Engine",
    description="Backend service to translate architectural images into structural code topology blueprints."
)

# Define the expected JSON payload schema
class BlueprintRequest(BaseModel):
    image_base64: str  # The raw Base64 string of the architecture sketch

# Point this to your external GPU server running Ollama
EXTERNAL_OLLAMA_URL = "http://<YOUR_EXTERNAL_SERVER_IP>:11434/api/generate"

@app.post("/vision/analyze")
async def process_blueprint(payload: BlueprintRequest):
    vision_prompt = (
        "Analyze this system architecture diagram or whiteboard sketch carefully. "
        "Identify all structural components (servers, gateways, databases, microservices) and how traffic flows between them. "
        "Convert this image into a clean text-based structural layout describing:\n"
        "1. Nodes (Name, Type/Technology if shown)\n"
        "2. Interconnections (Which component talks to which component, paths/routes used, ports)\n"
        "Do not write any infrastructure code or configuration scripts yet. Provide only the clear, detailed structural layout topology."
    )
    
    try:
        # Forward the request data directly to your remote GPU cluster
        response = requests.post(
            EXTERNAL_OLLAMA_URL,
            json={
                "model": "llava",  
                "prompt": vision_prompt,
                "images": [payload.image_base64],
                "stream": False
            },
            timeout=60  # Give the VLM plenty of time to process the layout matrix
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {response.text}")
            
        data = response.json()
        return {
            "status": "success",
            "topology": data.get("response", "")
        }
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="External GPU pipeline timed out during inference.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision processing failed: {str(e)}")
        
         # Import the official Ollama client

# ── Remote Server Configuration ───────────────────────────────────────────────

# CHANGE THIS to your external GPU server's actual IP address or domain name
# If you are using an SSH tunnel (e.g., ssh -L 11434:localhost:11434 user@server), leave it as localhost
EXTERNAL_OLLAMA_HOST = "http://http://10.22.39.192:11434:11434"

MODEL = "mistral:latest"
MAX_ITERATIONS = 30

SYSTEM_PROMPT = """You are an autonomous AI agent that accomplishes user-defined goals \
by planning and executing steps with available tools.

Approach each goal:
1. Understand what is being asked
2. Break it into concrete, executable steps
3. Use tools to carry out each step — read before you write
4. Verify results and correct course if something fails
5. Summarize what was accomplished when done

Be concise in your explanations. Prefer action over description. \
When you have achieved the goal, say so clearly."""

# ── Tool implementations ──────────────────────────────────────────────────────

def _read_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error: {exc}"


def _write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to '{path}'."
    except Exception as exc:
        return f"Error: {exc}"


def _list_directory(path: str = ".") -> str:
    try:
        entries = sorted(Path(path).iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        if not entries:
            return "(empty)"
        lines = []
        for e in entries:
            tag = "file" if e.is_file() else "dir "
            lines.append(f"[{tag}] {e.name}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error: {exc}"


def _run_python(code: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        parts = []
        if result.stdout.strip():
            parts.append(result.stdout.strip())
        if result.stderr.strip():
            parts.append(f"[stderr]\n{result.stderr.strip()}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30 s"
    except Exception as exc:
        return f"Error: {exc}"


def _run_shell(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        parts = []
        if result.stdout.strip():
            parts.append(result.stdout.strip())
        if result.stderr.strip():
            parts.append(f"[stderr]\n{result.stderr.strip()}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts) or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30 s"
    except Exception as exc:
        return f"Error: {exc}"


# ── Tool registry (Ollama Native Format) ──────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a local file. Always read a file before overwriting it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a local file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Destination file path"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and sub-folders inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list (defaults to current directory)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python snippet and return its stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
]


def _dispatch(name: str, inputs: dict) -> str:
    """Route a tool call to the correct implementation."""
    if name == "read_file":
        return _read_file(inputs["path"])
    if name == "write_file":
        return _write_file(inputs["path"], inputs["content"])
    if name == "list_directory":
        return _list_directory(inputs.get("path", "."))
    if name == "run_python":
        return _run_python(inputs["code"])
    if name == "run_shell":
        return _run_shell(inputs["command"])
    return f"Unknown tool: {name}"


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(goal: str) -> None:
    # Instantiate the client pointing to your external GPU engine
    client = Client(host=EXTERNAL_OLLAMA_HOST)

    print(f"\nGoal: {goal}")
    print(f"Targeting GPU Instance: {EXTERNAL_OLLAMA_HOST} [{MODEL}]")
    print("=" * 64)

    # Initialize standard chat format array
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal}
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n⚡ Iteration {iteration}/{MAX_ITERATIONS}")
        
        try:
            # Query Ollama with streaming enabled to output responses smoothly
            response_text = ""
            tool_calls = []
            
            response = client.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                stream=False # We use False here to ensure atomic verification of Tool Calls from Mistral
            )
            
            # Print assistant thought text if it exists
            if response.message.content:
                print(response.message.content)
                response_text = response.message.content
            
            # Extract tool choices chosen by Mistral
            if response.message.tool_calls:
                tool_calls = response.message.tool_calls
                
        except Exception as e:
            print(f"\n❌ Remote Ollama Connection Failed: {e}")
            print("Verify your IP routing, OLLAMA_HOST configuration, or SSH Tunnel status.")
            break

        # Append assistant's thoughts/actions to conversation state
        messages.append(response.message)

        # ── Termination condition check ─────────────────────────────────────────
        # If the model didn't ask for a tool, it assumes the task is complete
        if not tool_calls:
            print("\n🏁 Agent has finished processing adjustments.")
            break

        # ── Execute tool calls ────────────────────────────────────────────
        for call in tool_calls:
            function_info = call.function
            tool_name = function_info.name
            tool_args = function_info.inputs if hasattr(function_info, 'inputs') else function_info.arguments
            
            args_preview = json.dumps(tool_args, ensure_ascii=False)
            if len(args_preview) > 120:
                args_preview = args_preview[:117] + "..."
            print(f"\n  [tool] {tool_name}  {args_preview}")

            # Run execution block locally
            result = _dispatch(tool_name, tool_args)

            # Print tool response snippet
            preview = result if len(result) <= 300 else result[:297] + "..."
            for line in preview.splitlines():
                print(f"         {line}")

            # Return execution metric back to Ollama context history
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": result
            })

    else:
        print(f"\n[reached iteration limit of {MAX_ITERATIONS}]")

    print("\n" + "=" * 64)
    print("Done.\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        goal = " ".join(sys.argv[1:])
    else:
        print("Autonomous Local-GPU Agent Engine")
        print("Describe a goal and the agent will run tasks via remote inference.\n")
        goal = input("Goal: ").strip()
        if not goal or goal.lower() in {"quit", "exit", "q"}:
            return

    run_agent(goal)


if __name__ == "__main__":
    main()