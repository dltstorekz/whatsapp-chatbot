import os
from flask import Flask, request
from transformers import pipeline
import yaml
import importlib.util
import pathlib


def load_prompts():
    if not os.path.exists("prompts.yaml"):
        return {}
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_scripts():
    scripts = {}
    scripts_path = pathlib.Path("scripts")
    if scripts_path.exists():
        for file in scripts_path.glob("*.py"):
            module_name = f"scripts.{file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            scripts[file.stem] = mod
    return scripts


# Load HuggingFace model pipeline
nlp = pipeline("text-generation", model="gpt2")

app = Flask(__name__)
prompts = load_prompts()
scripts = load_scripts()


@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming = request.form.get('Body', '')
    # default response
    response_text = ""
    if incoming.startswith("/"):
        cmd, _, rest = incoming.partition(" ")
        if cmd in prompts:
            template = prompts[cmd]
            text = template.replace("{{input}}", rest)
            result = nlp(text, max_length=100, num_return_sequences=1)
            response_text = result[0]['generated_text']
        elif cmd[1:] in scripts:
            module = scripts[cmd[1:]]
            if hasattr(module, "handle"):
                response_text = module.handle(rest)
            else:
                response_text = "Handler not found in script."
        else:
            response_text = "Unknown command."
    else:
        result = nlp(incoming, max_length=100, num_return_sequences=1)
        response_text = result[0]['generated_text']
    return f"<Response><Message>{response_text}</Message></Response>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
