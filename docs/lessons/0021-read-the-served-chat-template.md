Read the served model's chat template before tuning around it: its defaults can be the biggest knob in the system.

The Flash-Next checkpoint's `chat_template.jinja` (HF RadixArk/Qwen3.8-Flash-Next-NVFP4) sets reasoning effort to
"xhigh" unless the request passes `reasoning_effort` in `chat_template_kwargs`, and that default prepends "think
carefully ... consider plausible alternatives" to the system prompt. It also takes `preserve_thinking` (false drops
the reasoning of past turns). Every public notebook that serves this model plays at xhigh because none passes the
kwarg, and about 81% of the model's output is reasoning in a time-bound run. We found it only by fetching and rendering
the template (research log 2026-09-23); the harness code and the serving flags never mention it.

How to apply: for any new served model, fetch its chat template, grep for kwargs (`enable_thinking`,
`reasoning_effort`, `preserve_thinking`, budgets), render it locally with jinja2 for each setting, and log what each
setting changes in the prompt before running an arm on it.
