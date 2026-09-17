"""ChatClient request shaping: where the reasoning effort goes depends on the served model family."""
from arc3.llm import ChatClient


class _Resp:
    status_code = 200

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class _Session:
    def __init__(self):
        self.bodies = []

    def post(self, url, headers, json, timeout):
        self.bodies.append(json)
        return _Resp()


def test_reasoning_effort_goes_to_the_chat_template_by_default():
    s = _Session()
    ChatClient(model="m", session=s).chat([{"role": "user", "content": "hi"}], reasoning_effort="xhigh", thinking=True)
    body = s.bodies[-1]
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"} and "reasoning_effort" not in body


def test_reasoning_effort_as_request_field_for_harmony_models():
    s = _Session()
    ChatClient(model="m", session=s, effort_in_request=True).chat([{"role": "user", "content": "hi"}], reasoning_effort="low")
    body = s.bodies[-1]
    assert body["reasoning_effort"] == "low" and "chat_template_kwargs" not in body
    # thinking on/off still travels as a template kwarg (Nemotron-style models)
    ChatClient(model="m", session=s, effort_in_request=True).chat([{"role": "user", "content": "hi"}], reasoning_effort="low", thinking=False)
    assert s.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}
