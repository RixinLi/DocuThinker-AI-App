import os
import logging
from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForQuestionAnswering, ORTModelForCausalLM, ORTModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config import MODEL_NAMES, TRANSLATION_MODELS, PROMPT_TEMPLATES
from .onnx_helper import check_onnx_model_exists, get_onnx_model_path

logger = logging.getLogger(__name__)

# Check if we should use ONNX models
os.environ["USE_ONNX"] = "TRUE" # 直接使用本地
# USE_ONNX = os.getenv("USE_ONNX", "false").lower() == "true"
USE_ONNX = True
# print("using USE_ONNX =",USE_ONNX)s


def load_models():
    models = {}

    # Summarizer chain
    try:
        logger.info("Loading summarizer pipeline...")
        summarizer_chain = _create_seq2seq_chain(
            model_name=MODEL_NAMES["summarizer"],
            onnx_subdir="summarizer",
            prompt_template=PROMPT_TEMPLATES["summarization"]
        )
        models["summarizer_chain"] = summarizer_chain
    except Exception as e:
        logger.exception("Error loading summarization chain: %s", e)
        raise e

    # # QA chain
    try:
        logger.info("Loading QA pipeline...")
        qa_chain = _create_qa_chain(
            model_name=MODEL_NAMES["qa"],
            onnx_subdir="qa",
            prompt_template=PROMPT_TEMPLATES["qa"]
        )
        models["qa_chain"] = qa_chain
    except Exception as e:
        logger.exception("Error loading QA chain: %s", e)
        raise e

    # # Discussion chain
    try:
        logger.info("Loading discussion generator pipeline...")
        discussion_chain = _create_textgen_chain(
            model_name=MODEL_NAMES["discussion"],
            onnx_subdir="discussion",
            prompt_template=PROMPT_TEMPLATES["discussion"]
        )
        models["discussion_chain"] = discussion_chain
    except Exception as e:
        logger.exception("Error loading discussion chain: %s", e)
        raise e

    # # RAG chain
    # try:
    #     logger.info("Loading RAG pipeline...")
    #     rag_chain = _create_text2text_chain(
    #         model_name=MODEL_NAMES["rag"],
    #         onnx_subdir="rag",
    #         prompt_template=PROMPT_TEMPLATES["rag"]
    #     )
    #     models["rag_chain"] = rag_chain
    # except Exception as e:
    #     logger.exception("Error loading RAG chain: %s", e)
    #     raise e

    # # Topic extractor
    try:
        logger.info("Loading topic extraction pipeline...")
        models["topic_extractor"] = pipeline("zero-shot-classification", model=MODEL_NAMES["topic_extractor"])
    except Exception as e:
        logger.exception("Error loading topic extractor: %s", e)
        raise e

    # # Sentiment analyzer
    try:
        logger.info("Loading sentiment analyzer pipeline...")
        sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"
        if USE_ONNX:
            onnx_path = get_onnx_model_path("sentiment")
            if check_onnx_model_exists(onnx_path):
                tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
                models["sentiment_analyzer"] = pipeline(
                    "sentiment-analysis",
                    model=onnx_path,
                    tokenizer=tokenizer,
                    framework="onnxruntime"
                )
            else:
                models["sentiment_analyzer"] = pipeline("sentiment-analysis", model=sentiment_model)
        else:
            models["sentiment_analyzer"] = pipeline("sentiment-analysis", model=sentiment_model)
    except Exception as e:
        logger.exception("Error loading sentiment analyzer: %s", e)
        raise e

    return models

def load_translation_model(target_lang):
    """
    Dynamically load a translation pipeline for the specified target language.
    """
    if target_lang not in TRANSLATION_MODELS:
        raise ValueError(f"Translation model for language '{target_lang}' is not configured.")

    model_name = TRANSLATION_MODELS[target_lang]
    task_name = f"translation_en_to_{target_lang}"
    logger.info("Loading translator model for language '%s'...", target_lang)

    if USE_ONNX:
        from transformers import AutoTokenizer
        onnx_path = get_onnx_model_path("translation", target_lang)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if check_onnx_model_exists(onnx_path):
            translator = pipeline(
                task_name,
                model=onnx_path,
                tokenizer=tokenizer,
                framework="onnxruntime"
            )
        else:
            translator = pipeline(task_name, model=model_name)
    else:
        translator = pipeline(task_name, model=model_name)
    return translator

def _create_seq2seq_chain(model_name, onnx_subdir, prompt_template):
    """
    Creates a seq2seq chain (e.g. summarization).
    """
    if USE_ONNX:
        onnx_path = get_onnx_model_path(onnx_subdir)
        if check_onnx_model_exists(onnx_path):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe = pipeline("summarization", model=onnx_path, tokenizer=tokenizer, framework="onnxruntime")
        else:
            pipe = pipeline("summarization", model=model_name)
    else:
        pipe = pipeline("summarization", model=model_name)

    llm = HuggingFacePipeline(pipeline=pipe)
    template = PromptTemplate(input_variables=["text"], template=prompt_template)
    # 新写法：用 RunnableSequence
    chain = template | llm
    return chain

def _create_qa_chain(model_name, onnx_subdir, prompt_template):
    """
    Creates a QA chain.
    """
    if USE_ONNX:
        onnx_path = get_onnx_model_path(onnx_subdir)
        if check_onnx_model_exists(onnx_path):
            # 用 Optimum 加载 ONNX 模型
            model = ORTModelForQuestionAnswering.from_pretrained(
                onnx_path, file_name="model.onnx"
            )
            tokenizer = AutoTokenizer.from_pretrained(onnx_path)

            # 补充 name_or_path 属性，避免 HuggingFacePipeline 报错
            if not hasattr(model, "name_or_path"):
                model.name_or_path = onnx_path  

            pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        else:
            # 回退到普通 PyTorch 模型
            pipe = pipeline("question-answering", model=model_name)
    else:
        pipe = pipeline("question-answering", model=model_name)

    llm = HuggingFacePipeline(pipeline=pipe)
    template = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    # 新写法：用 RunnableSequence
    chain = template | llm
    return chain

def _create_textgen_chain(model_name, onnx_subdir, prompt_template):
    """
    Creates a text-generation chain (e.g. for discussion).
    """
    if USE_ONNX:
        onnx_path = get_onnx_model_path(onnx_subdir)
        if check_onnx_model_exists(onnx_path):
            # 判断是 causal LM 还是 seq2seq LM
            try:
                model = ORTModelForCausalLM.from_pretrained(onnx_path, file_name="model.onnx")
                
            except Exception:
                model = ORTModelForSeq2SeqLM.from_pretrained(onnx_path, file_name="model.onnx")

            tokenizer = AutoTokenizer.from_pretrained(onnx_path)

            # 补充 name_or_path，避免 HuggingFacePipeline 报错
            if not hasattr(model, "name_or_path"):
                model.name_or_path = onnx_path
            
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        else:
            pipe = pipeline("text-generation", model=model_name)
    else:
        pipe = pipeline("text-generation", model=model_name)

    # 这里暂时先不使用缓存，但是discussion模型还是需要缓存更为实际
    pipe.model.generation_config.use_cache = False
    llm = HuggingFacePipeline(pipeline=pipe)
    template = PromptTemplate(input_variables=["text"], template=prompt_template)
    # 新写法：用 RunnableSequence
    chain = template | llm
    return chain

def _create_text2text_chain(model_name, onnx_subdir, prompt_template):
    """
    Creates a text2text-generation chain (e.g. RAG).
    """
    if USE_ONNX:
        onnx_path = get_onnx_model_path(onnx_subdir)
        if check_onnx_model_exists(onnx_path):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            pipe = pipeline("text2text-generation", model=onnx_path, tokenizer=tokenizer, framework="onnxruntime")
        else:
            pipe = pipeline("text2text-generation", model=model_name)
    else:
        pipe = pipeline("text2text-generation", model=model_name)

    llm = HuggingFacePipeline(pipeline=pipe)
    template = PromptTemplate(input_variables=["text"], template=prompt_template)
    return LLMChain(llm=llm, prompt=template)


def main():
    models = load_models()
    # print(models["summarizer_chain"])
    # print(models["qa_chain"])
    print(models["discussion_chain"])
    return

if __name__ == "__main__":
    main()