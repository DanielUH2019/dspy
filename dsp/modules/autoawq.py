from typing import Any, Optional

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

from dsp.modules.lm import LM


def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        elif k == "max_tokens":
            hf_kwargs["max_new_tokens"] = v
        elif k == "model":
            pass
        else:
            hf_kwargs[k] = v

    return hf_kwargs


class AutoAwq(LM):
    def __init__(
        self,
        model: str,
        quant_path: str,
        quant_file: str,
        streamer: Optional[TextStreamer],
    ):
        super().__init__(model)
        self.loaded_model = AutoAWQForCausalLM.from_quantized(
            quant_path, quant_file, fuse_layers=False, safetensors=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            quant_path, trust_remote_code=True
        )
        self.streamer = streamer
        self.drop_prompt_from_output = True
        self.history: list[dict[str, Any]] = []

    def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}

        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.cuda()

        outputs = self.loaded_model.generate(tokens, streamer=self.streamer, **kwargs)
        if self.drop_prompt_from_output:
            input_length = tokens.shape[1]
            outputs = outputs[:, input_length:]

        completions = [
            {"text": c}
            for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ]
        response = {
            "prompt": prompt,
            "choices": completions,
        }

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    def request(self, prompt: str, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def get_choice_text(self, choice) -> str:
        return choice["text"]

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from GPT-3.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs)
        return [c["text"] for c in response["choices"]]
