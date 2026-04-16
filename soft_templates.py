from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class TemplateSpec:
    name: str
    with_system: List[str]
    without_system: List[str]
    mmlu_output_prefix: str = "The correct answer is"


TEMPLATES: Dict[str, TemplateSpec] = {
    "vicuna": TemplateSpec(
        name="vicuna",
        with_system=["{system_prompt}", "\n\nUSER:", " {prompt}", "\nASSISTANT:"],
        without_system=["USER:", " {prompt}", "\nASSISTANT:"],
    ),
    "llama3": TemplateSpec(
        name="llama3",
        with_system=[
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "{system_prompt}",
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
            "{prompt}",
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ],
        without_system=[
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
            "{prompt}",
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        ],
    ),
    "llama2": TemplateSpec(
        name="llama2",
        with_system=["<s>[INST] <<SYS>>\n", "{system_prompt}", "\n<</SYS>>\n\n", "{prompt}", " [/INST]\n"],
        without_system=["<s>[INST] <<SYS>>\n\n<</SYS>>\n\n", "{prompt}", " [/INST]\n"],
    ),
    "mistral": TemplateSpec(
        name="mistral",
        with_system=["<s>[INST] <<SYS>>\n", "{system_prompt}", "\n<</SYS>>\n\n", "{prompt}", " [/INST]\n"],
        without_system=["<s>[INST] <<SYS>>\n\n<</SYS>>\n\n", "{prompt}", " [/INST]\n"],
    ),
    "deepseek": TemplateSpec(
        name="deepseek",
        with_system=["User:", " {system_prompt}", " {prompt}", "\nAssistant:"],
        without_system=["User:", " {prompt}", "\nAssistant:"],
    ),
    "openchat": TemplateSpec(
        name="openchat",
        with_system=["<|system|>\n", "{system_prompt}", "\n<|user|>\n", "{prompt}", "\n<|assistant|>\n"],
        without_system=["<|system|>\n\n<|user|>\n", "{prompt}", "\n<|assistant|>\n"],
    ),
}


def infer_template_family(model_name: str) -> str | None:
    model_name_l = model_name.lower()
    if "vicuna" in model_name_l:
        return "vicuna"
    if "llama-3" in model_name_l or "llama3" in model_name_l:
        return "llama3"
    if "llama-2" in model_name_l or "llama2" in model_name_l:
        return "llama2"
    if "mistral" in model_name_l:
        return "mistral"
    if "deepseek" in model_name_l:
        return "deepseek"
    if "openchat" in model_name_l:
        return "openchat"
    return None


def get_template(model_name: str, template_family: str = "auto") -> TemplateSpec:
    family = infer_template_family(model_name) if template_family == "auto" else template_family
    if family is None:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise ValueError(
            "No chat template is defined for this model. "
            "Add the model template in soft_templates.py and then rerun. "
            f"Currently available template families: {available}"
        )
    if family not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise ValueError(
            f"Unknown template family '{family}'. "
            f"Choose one of: {available}, or add a new one in soft_templates.py."
        )
    return TEMPLATES[family]


def available_template_families() -> List[str]:
    return sorted(TEMPLATES.keys())


def choose_format_list(
    template: TemplateSpec,
    include_system_prompt: bool,
    include_mmlu_prefix: bool = False,
) -> List[str]:
    """
    Return the prompt chunks for the selected template.

    include_system_prompt=True means the rendered prompt should contain the
    template's system-role section. This is required for DDPO's `sys_prompt`
    mode and is also useful when the user wants to provide a fixed textual
    system prompt in prefix/suffix mode.
    """
    parts = list(template.with_system if include_system_prompt else template.without_system)
    if include_mmlu_prefix:
        parts[-1] = parts[-1] + template.mmlu_output_prefix
    return parts


def render_prompt_from_chunks(format_list: Sequence[str], prompt: str, system_prompt: str = "") -> str:
    return "".join(format_list).replace("{system_prompt}", system_prompt).replace("{prompt}", prompt)
