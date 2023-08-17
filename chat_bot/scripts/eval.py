import sys

sys.path.append("./")
from chat_bot.neural_chat.e2emodel.e2e_lora_model import E2ELoRA
from chat_bot.neural_chat.conversation import get_conv_template
from transformers import GenerationConfig
import argparse
import torch
import random
import json
from typing import Dict


def rollout(
    model: E2ELoRA,
    scenario: Dict,
    gen_config: GenerationConfig,
    conv_template_name: str,
):
    conv = get_conv_template(conv_template_name)
    conv.load_dict(scenario)
    conv.messages = []
    print(conv.get_scenario())
    while True:
        user_input = input()
        if user_input == "quit":
            break
        conv.append_message("구매자", user_input)
        conv.append_message("판매자", "")
        model_response = model.generate(conv, gen_config)
        print(model_response)
        if model_response.startswith("##<"):
            break
        conv.update_last_message(model_response)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./data/sample_data.json")
    parser.add_argument(
        "--model_checkpoint_path", default="ggul-tiger/negobot_361_4bit_v1"
    )
    parser.add_argument("--conv-template-name", default="v2")
    parser.add_argument("--num-rollouts", type=int, default=30)
    args = parser.parse_args()

    gen_config = GenerationConfig(
        max_new_tokens=128,
        use_cahce=True,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.85,
        num_beams=3,
        temperature=0.9,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = E2ELoRA(args.model_checkpoint_path, device=device, do_quantize=True)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(args.num_rollouts):
        print(f"rollout #{i + 1}")
        scenario = random.choice(data)
        rollout(model, scenario, gen_config, args.conv_template_name)
