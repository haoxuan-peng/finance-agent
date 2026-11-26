"""ms-swift external plugin: preserve independent loss flags in parallel calls.

Load with --external_plugins /absolute/path/to/this/file.py. Native Qwen3.5
formatting remains authoritative; only its loss annotations are split. Requires
the swift backend and a version with per-segment assistant loss support.
"""

from __future__ import annotations

import inspect


def split_mixed_tool_batches(template, inputs) -> None:
    messages = inputs.messages
    i = 0
    while i < len(messages):
        if messages[i]["role"] != "tool_call":
            i += 1
            continue
        end = i + 1
        while end < len(messages) and messages[end]["role"] == "tool_call":
            end += 1
        batch = messages[i:end]
        settings = {(m.get("loss"), m.get("loss_scale")) for m in batch}
        if len(settings) <= 1:
            i = end
            continue
        if template.template_backend != "swift":
            raise RuntimeError("Per-call masking requires --template_backend swift")
        agent = template.agent_template
        agent.template_meta = template.template_meta
        whole = agent._format_tool_calls(batch)
        chunks = []
        cursor = 0
        for message in batch:
            single = agent._format_tool_calls([message])
            start = whole.find(single, cursor)
            if start < 0 or whole[cursor:start].strip():
                raise RuntimeError(
                    "Agent template cannot be split without changing its rendered calls"
                )
            stop = start + len(single)
            chunks.append(whole[cursor:stop])
            cursor = stop
        if whole[cursor:].strip():
            raise RuntimeError("Unexpected batch suffix in agent template")
        chunks[-1] += whole[cursor:]
        previous = messages[i - 1] if i else None
        prefixed = agent._add_tool_call_prefix(whole, previous)
        if not prefixed.endswith(whole):
            raise RuntimeError(
                "Agent template changes call bodies while adding the prefix"
            )
        chunks[0] = prefixed[: len(prefixed) - len(whole)] + chunks[0]
        if "".join(chunks) != prefixed:
            raise RuntimeError("Rendered tool-call bytes changed")
        replacements = []
        for message, chunk in zip(batch, chunks):
            replacement = {"role": "assistant", "content": chunk}
            for key in ["loss", "loss_scale"]:
                if key in message:
                    replacement[key] = message[key]
            replacements.append(replacement)
        messages[i:end] = replacements
        i = end


def install() -> None:
    from swift.template import Template

    original = Template._preprocess_tool_call
    if getattr(original, "_finance_per_call_mask", False):
        return
    merge_source = inspect.getsource(Template._swift_prepare_inputs)
    if "loss_scale" not in merge_source or "cur_val" not in merge_source:
        raise RuntimeError(
            "This ms-swift version lacks independent loss flags for merged assistant segments; upgrade ms-swift"
        )

    def preprocess(self, inputs):
        split_mixed_tool_batches(self, inputs)
        return original(self, inputs)

    preprocess._finance_per_call_mask = True
    Template._preprocess_tool_call = preprocess


# A normal import installs the plugin; unit tests can load the helper without
# importing the optional training stack via runpy.run_path(..., run_name=...).
if __name__ != "_test_swift_per_call_loss":
    install()
