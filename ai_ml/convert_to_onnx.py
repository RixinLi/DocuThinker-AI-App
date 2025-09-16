import os
import subprocess
import sys


# 1. 普通模型导出
def run_conversion(model_name, task, output_dir):
    """
    Runs the ONNX export command for a given model, task, and output directory.
    """
    command = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_name,
        "--task", task,
        "--use_cache",   # 帮助hugging face使用缓存
        output_dir
    ]
    print("Running command:", " ".join(command))
    subprocess.run(command, check=True)

# 2. RAG专用导出
def export_rag(model_name, encoder_dir, generator_dir):
    from transformers import RagTokenForGeneration
    import torch
    # 加载RAG复合模型
    rag = RagTokenForGeneration.from_pretrained(model_name)
    rag.eval()

    # 建立dir
    os.makedirs(encoder_dir, exist_ok=True)
    os.makedirs(generator_dir, exist_ok=True)

    #导出question_encoder
    torch.onnx.export(
        rag.question_encoder,
        (torch.randint(0,100,(1,16)),torch.ones(1,16)),
        f"{encoder_dir}/rag_question_encoder.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["pooler_output"],
        dynamic_axes={"input_ids":{0:"batch",1:"sequence"}},
        opset_version=14
    )

    #导出generator
    torch.onnx.export(
        rag.generator,
        (
            torch.randint(0, 100, (1, 16)),
            torch.ones(1, 16),
            torch.randint(0, 100, (1, 16)),
            torch.ones(1, 16)
        ),
        f"{generator_dir}/rag_generator.onnx",
        input_names=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        opset_version=14
    )
    # print("rag encoder and generator models saved!")

# # 3. 翻译模型（MarianMT）专用导出
def export_translation(model_name, output_dir):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch, os

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    # 保存 tokenizer 到同一目录
    tokenizer.save_pretrained(output_dir)

    example = tokenizer("Hello world", return_tensors="pt")
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

    torch.onnx.export(
        model,
        (example["input_ids"], example["attention_mask"], decoder_input_ids),
        f"{output_dir}/model.onnx",
        input_names=["input_ids", "attention_mask", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "decoder_input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14
    )


def main():
    # Create base directories for ONNX models
    base_dirs = [
        "onnx_models/summarizer",
        "onnx_models/qa",
        "onnx_models/discussion",
        # "onnx_models/rag", #暂时跳过RAG
        "onnx_models/sentiment",
    ]

    # Translation model mappings (language code to model name)
    translations = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "de": "Helsinki-NLP/opus-mt-en-de",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "it": "Helsinki-NLP/opus-mt-en-it",
        "zh": "Helsinki-NLP/opus-mt-en-zh",
    }

    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)

    # Create directories for translation models
    for lang in translations.keys():
        os.makedirs(f"onnx_models/translation/{lang}", exist_ok=True)

    try:
        # ================ Run Conversion on optimum.exporters models ========================= 
        # Convert Summarizer model
        run_conversion("facebook/bart-large-cnn", "summarization", "onnx_models/summarizer")

        # Convert QA model
        run_conversion("distilbert-base-cased-distilled-squad", "question-answering", "onnx_models/qa")

        # Convert Discussion Generator model (using GPT-2 for text generation)
        run_conversion("gpt2", "text-generation", "onnx_models/discussion")

        # Convert Sentiment Analysis model
        run_conversion("distilbert-base-uncased-finetuned-sst-2-english", "sequence-classification", "onnx_models/sentiment")


        # ================ Run Rag combined model into two models ========================= 
        # Convert RAG model (for text2text-generation)
        # 分开导出复合rag模型的两个模型
        export_rag("facebook/rag-token-nq", "onnx_models/rag/question_encode", "onnx_models/rag/generator")        


        # ================ Run Translations with models and tokenizers ========================= 
        # 批量调用 export_translation
        for lang, model_name in translations.items():
            output_dir = f"onnx_models/translation/{lang}"
            export_translation(model_name, output_dir)

        print("All models have been successfully converted to ONNX format!")
    except subprocess.CalledProcessError as e:
        print("An error occurred during ONNX conversion:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()

