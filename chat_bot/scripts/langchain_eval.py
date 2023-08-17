import torch
import argparse
import json
import random

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TextGenerationPipeline,
)
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
)
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate


def rollout(
    conversation: ConversationChain,
    scenario: dict,
):
    conversation.memory.clear()
    scenario.pop("events")
    conversation.prompt = conversation.prompt.partial(**scenario)
    while True:
        user_input = input()
        if user_input == "quit":
            break
        response = conversation(user_input, return_only_outputs=True)
        print(response)


def load_pipeline(model_name_or_path: str) -> TextGenerationPipeline:
    peft_config = PeftConfig.from_pretrained(model_name_or_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model = PeftModelForCausalLM.from_pretrained(
        model, model_name_or_path, torch_dtype=torch.float16
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        early_stopping=True,
        do_sample=True,
        top_k=50,
        top_p=0.85,
        num_beams=3,
        temperature=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    return pipe


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./data/sample_data.json")
    parser.add_argument(
        "--adapter-name-or-path", default="ggul-tiger/negobot_361_4bit_v1"
    )
    parser.add_argument("--conv-template-name", default="v2")
    parser.add_argument("--num-rollouts", type=int, default=30)
    args = parser.parse_args()

    prompt = PromptTemplate.from_file(
        f"./chat_bot/neural_chat/templatates/{args.conv_template_name}.txt",
        input_variables=["title", "description", "price", "history", "user_input"],
    )
    prompt = prompt.partial(title="", description="", price="")

    memory = ConversationBufferWindowMemory(
        k=5,
        human_prefix="구매자",
        ai_prefix="판매자",
        memory_key="history",
    )

    pipe = load_pipeline(args.adapter_name_or_path)
    llm = HuggingFacePipeline(pipeline=pipe)
    conversation = ConversationChain(
        llm=llm, prompt=prompt, memory=memory, verbose=True, input_key="user_input"
    )

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i in range(args.num_rollouts):
        print(f"rollout #{i + 1}")
        scenario = random.choice(data)
        print(
            *[f"{key}: {scenario[key]}" for key in ["title", "description", "price"]],
            sep="\n",
        )
        rollout(conversation, scenario)


if __name__ == "__main__":
    main()
