
import subprocess, tempfile, os, textwrap, sys

def python_exec(code: str, timeout_sec: int = 8) -> str:
    code = textwrap.dedent(code)
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "snippet.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            out = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=timeout_sec)
            stdout = out.stdout.strip()
            stderr = out.stderr.strip()
            if out.returncode != 0:
                return f"[python_exec ERROR]\n{stderr}"
            return stdout or "[python_exec OK: no output]"
        except subprocess.TimeoutExpired:
            return "[python_exec TIMEOUT]"
