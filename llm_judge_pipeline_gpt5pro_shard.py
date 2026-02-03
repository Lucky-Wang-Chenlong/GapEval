#!/usr/bin/env python
import argparse
import base64
import json
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path

import requests
import yaml
from jinja2 import Environment, StrictUndefined
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DATA_ROOT_DEFAULT = "./"
PROMPT_DIR_DEFAULT = "utils/prompt_templates"
ERROR_DIR_DEFAULT = ""
MODEL_DEFAULT = ""
# API_KEY_DEFAULT = "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"
API_KEY_DEFAULT = ""
PROXY_ENV_KEYS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
]

MAX_TOKENS_DEFAULT = 99999
API_WAIT_SECONDS = 0.5


PROMPT_MAP = {
    "Multi Hop": {
        "text": "multi_hop_text_eval.yaml",
        "image": "multi_hop_image_eval.yaml",
    },
    "physics": {
        "text": "physics_text_eval.yaml",
        "image": "physics_image_eval.yaml",
    },
    "rule_based": {
        "text": "rule_based_text_eval.yaml",
        "image": "rule_based_image_eval.yaml",
    },
}


def ensure_dir(path_value):
    Path(path_value).mkdir(parents=True, exist_ok=True)


def clear_proxy_env():
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def log_error(error_log_path, model_dir_path, error_message):
    ensure_dir(Path(error_log_path).parent)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(error_log_path, "a", encoding="utf-8") as error_file:
        error_file.write(f"{timestamp}\t{model_dir_path}\t{error_message}\n")


def normalize_log_value(value, max_len=10000):
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False)
        except Exception:
            value = str(value)
    value = value.replace("\n", "\\n").replace("\r", "\\r")
    if len(value) > max_len:
        return value[:max_len] + "...[truncated]"
    return value


def log_error_with_raw(error_log_path, model_dir_path, error_message, raw_response):
    ensure_dir(Path(error_log_path).parent)
    timestamp = datetime.now().isoformat(timespec="seconds")
    error_message = normalize_log_value(error_message) or ""
    raw_response = normalize_log_value(raw_response) or ""
    with open(error_log_path, "a", encoding="utf-8") as error_file:
        error_file.write(f"{timestamp}\t{model_dir_path}\t{error_message}\t{raw_response}\n")


class ApiResponseError(Exception):
    def __init__(self, message, raw_response):
        super().__init__(message)
        self.raw_response = raw_response


def parse_data_id(data_point_dir):
    name = Path(data_point_dir).name
    match = re.match(r"data_(\d+)$", name)
    if not match:
        return None
    return int(match.group(1))


def get_mod_group(data_point_dir):
    data_id = parse_data_id(data_point_dir)
    if data_id is None:
        return None
    return data_id % 3


def get_error_log_path_for_mod(mod_value):
    if mod_value in (0, 1, 2):
        return str(Path(ERROR_DIR_DEFAULT) / f"error{mod_value + 1}.txt")
    if mod_value == 3:
        return str(Path(ERROR_DIR_DEFAULT) / "error4.txt")
    return str(Path(ERROR_DIR_DEFAULT) / "error.txt")


def encode_image_to_data_url(image_path):
    image_path = Path(image_path)
    suffix = image_path.suffix.lower()
    mime_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_image}"


def load_yaml_prompt(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return {
        "system_prompt": data.get("system_prompt", ""),
        "template": data.get("template", ""),
        "jinja_args": data.get("jinja_args", []),
    }


def render_template(template_str, **kwargs):
    environment = Environment(undefined=StrictUndefined)
    template = environment.from_string(template_str)
    return template.render(**kwargs)


def parse_json_maybe(text_value):
    try:
        return json.loads(text_value)
    except Exception:
        pass
    start_index = text_value.find("{")
    end_index = text_value.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        try:
            return json.loads(text_value[start_index : end_index + 1])
        except Exception:
            return None
    return None


def extract_score_value(parsed):
    if not isinstance(parsed, dict):
        return None
    score = parsed.get("score")
    if score is None:
        score = parsed.get("judge_result")
    if score is None:
        score = parsed.get("judeg_result")
    return score


def is_positive_score(score):
    if isinstance(score, bool):
        return score
    if isinstance(score, (int, float)):
        return score == 1
    if isinstance(score, str):
        return score.strip().lower() in {"1", "true", "yes"}
    return False


def build_messages(system_prompt, template_str, template_vars, image_paths=None):
    user_text = render_template(template_str, **template_vars)
    if image_paths:
        content_parts = [{"type": "text", "text": user_text}]
        for image_path in image_paths:
            if image_path:
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": encode_image_to_data_url(image_path)}}
                )
        user_content = content_parts
    else:
        user_content = user_text
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def create_requests_session(use_proxy):
    session = requests.Session()
    if not use_proxy:
        session.trust_env = False
    return session


def wait_before_request():
    if API_WAIT_SECONDS > 0:
        time.sleep(API_WAIT_SECONDS)


def call_text_model(session, api_key, model_name, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "reasoning": {"enabled": True},
        "max_tokens": MAX_TOKENS_DEFAULT,
    }
    wait_before_request()
    response = session.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    response.raise_for_status()
    try:
        response_json = response.json()
    except Exception:
        raise ApiResponseError("invalid_json_response", response.text)
    if "choices" not in response_json:
        raise ApiResponseError("missing_choices_in_response", response_json)
    assistant_message = response_json["choices"][0]["message"]

    followup_messages = list(messages)
    followup_messages.append(
        {
            "role": "assistant",
            "content": assistant_message.get("content"),
            "reasoning_details": assistant_message.get("reasoning_details"),
        }
    )
    followup_messages.append({"role": "user", "content": "Are you sure? Think carefully."})
    payload2 = {
        "model": model_name,
        "messages": followup_messages,
        "reasoning": {"enabled": True},
        "max_tokens": MAX_TOKENS_DEFAULT,
    }
    wait_before_request()
    response2 = session.post(url, headers=headers, data=json.dumps(payload2), timeout=120)
    response2.raise_for_status()
    try:
        response2_json = response2.json()
    except Exception:
        raise ApiResponseError("invalid_json_response_followup", response2.text)
    if "choices" not in response2_json:
        raise ApiResponseError("missing_choices_in_response_followup", response2_json)
    return response2_json["choices"][0]["message"]["content"]


def call_image_model(session, api_key, model_name, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model_name, "messages": messages, "max_tokens": MAX_TOKENS_DEFAULT}
    wait_before_request()
    response = session.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    try:
        payload = response.json()
    except Exception:
        raise ApiResponseError("invalid_json_response", response.text)
    if "choices" not in payload:
        raise ApiResponseError("missing_choices_in_response", payload)
    return payload["choices"][0]["message"]["content"]


def resolve_prompt_image(data_point_dir, image_value):
    if not image_value:
        return None
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = Path(data_point_dir) / image_value
    return image_path if image_path.exists() else None


def read_prompt_json(data_point_dir):
    prompt_path = Path(data_point_dir) / "prompt.json"
    with open(prompt_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        if not data or not isinstance(data[0], dict):
            raise ValueError("prompt.json list is empty or not a list of objects")
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError("prompt.json must be an object or a list of objects")
    category = data.get("category")
    reference = data.get("reference")
    subcategory = data.get("subcategory")
    problem_text = data.get("und_prompt") or data.get("problem")
    gen_prompt = data.get("gen_prompt")
    problem_image = data.get("img")
    expect_image = data.get("expect_image")
    if isinstance(reference, list):
        reference_text = "\n".join([str(item) for item in reference])
    elif reference is None:
        reference_text = ""
    else:
        reference_text = str(reference)
    return {
        "category": category,
        "subcategory": subcategory,
        "reference_text": reference_text,
        "problem_text": problem_text,
        "gen_prompt": gen_prompt,
        "problem_image_value": problem_image,
        "expect_image_value": expect_image,
    }


def get_prompt_config(category, subcategory):
    if category == "counting":
        is_animal = subcategory == "Animals Counting"
        return {
            "text": "counting_text_animal_eval.yaml" if is_animal else "counting_text_object_eval.yaml",
            "image": "counting_image_animal_eval.yaml" if is_animal else "counting_image_object_eval.yaml",
        }
    if category == "reasoning":
        if subcategory == "draw":
            return {
                "text": "draw_reasoning_text_eval.yaml",
                "image": "draw_reasoning_image_eval.yaml",
            }
        if subcategory == "mmlu/mmmu":
            return {
                "text": "mmmu_mmlu_reasoning_choice_text_eval.yaml",
                "image": "mmmu_mmlureasoning_choice_image_eval.yaml",
                "balance": "mmmu_mmlu_reasoning_image_text_balance.yaml",
            }
        if subcategory == "puzzle":
            return {
                "text": "reasoning_choice_text_eval.yaml",
                "image": "reasoning_choice_image_eval.yaml",
                "balance": "reasoning_image_text_balance.yaml",
            }
    return PROMPT_MAP.get(category)


def find_reference_image(data_point_dir):
    for folder_name in ["ref", "image", "inage"]:
        image_dir = Path(data_point_dir) / folder_name
        if not image_dir.exists():
            continue
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if images:
            return images[0]
    return None


def find_ref_only_image(data_point_dir):
    image_dir = Path(data_point_dir) / "ref"
    if not image_dir.exists():
        return None
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if images:
        return images[0]
    return None


def find_problem_image(data_point_dir):
    for folder_name in ["image", "inage"]:
        image_dir = Path(data_point_dir) / folder_name
        if not image_dir.exists():
            continue
        images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if images:
            return images[0]
    return None


def list_model_dirs(data_point_dir):
    ignore_names = {"image", "inage", "ref", "lumoo"}
    data_point_dir = Path(data_point_dir)
    model_dirs = []
    for item in data_point_dir.iterdir():
        if item.is_dir() and item.name not in ignore_names and not item.name.startswith("."):
            if item.name.startswith("output_") and not item.name.endswith("1"):
                continue
            model_dirs.append(item)
    return sorted(model_dirs)


def load_generated_text(model_dir):
    text_path = Path(model_dir) / "text.json"
    if not text_path.exists():
        return None
    with open(text_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("output_text")


def list_generated_images(model_dir):
    images = []
    for item in Path(model_dir).iterdir():
        if item.is_file() and item.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            images.append(item)
    return sorted(images)


def should_skip_model_dir(model_dir):
    result_path = Path(model_dir) / "new_res.json"
    if not result_path.exists():
        return False
    try:
        data = json.load(open(result_path, "r", encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    has_error = False
    has_parsed = False
    has_skipped = False

    text_eval = data.get("text_eval")
    if isinstance(text_eval, dict):
        if text_eval.get("status") == "error":
            has_error = True
        if text_eval.get("status") == "skipped":
            has_skipped = True
        if text_eval.get("parsed"):
            has_parsed = True

    image_eval = data.get("image_eval")
    if isinstance(image_eval, list):
        for item in image_eval:
            if not isinstance(item, dict):
                continue
            if item.get("status") == "error":
                has_error = True
            if item.get("status") == "skipped":
                has_skipped = True
            if item.get("parsed"):
                has_parsed = True

    return (has_parsed or has_skipped) and not has_error


def load_existing_status(model_dir):
    result_path = Path(model_dir) / "new_res.json"
    if not result_path.exists():
        return None, None
    try:
        data = json.load(open(result_path, "r", encoding="utf-8"))
    except Exception:
        return None, None
    if not isinstance(data, dict):
        return None, None
    text_status = None
    image_has_error = False
    image_has_any = False
    text_eval = data.get("text_eval")
    if isinstance(text_eval, dict):
        text_status = text_eval.get("status")
    image_eval = data.get("image_eval")
    if isinstance(image_eval, list):
        for item in image_eval:
            if not isinstance(item, dict):
                continue
            image_has_any = True
            if item.get("status") == "error":
                image_has_error = True
    return text_status, (image_has_error if image_has_any else None)


def load_existing_result(model_dir):
    result_path = Path(model_dir) / "new_res.json"
    if not result_path.exists():
        return None
    try:
        data = json.load(open(result_path, "r", encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def extract_existing_text_score(existing_result):
    if not isinstance(existing_result, dict):
        return None, None
    text_eval = existing_result.get("text_eval")
    if not isinstance(text_eval, dict):
        return None, None
    parsed = text_eval.get("parsed")
    if parsed is None:
        raw_response = text_eval.get("raw_response")
        if isinstance(raw_response, str):
            parsed = parse_json_maybe(raw_response)
    score = extract_score_value(parsed)
    return score, parsed


def build_existing_image_eval_map(existing_image_eval):
    image_map = {}
    if not isinstance(existing_image_eval, list):
        return image_map
    for item in existing_image_eval:
        if not isinstance(item, dict):
            continue
        generated_image = item.get("generated_image")
        if not generated_image:
            continue
        eval_type = item.get("eval_type")
        image_map[(generated_image, eval_type)] = item
    return image_map


def should_keep_image_eval(existing_item):
    return isinstance(existing_item, dict) and existing_item.get("status") != "error"


def plan_model_work(existing_result, prompt_info):
    category = prompt_info.get("category")
    subcategory = prompt_info.get("subcategory")
    text_status = None
    image_has_error = False
    image_has_any = False

    if isinstance(existing_result, dict):
        text_eval = existing_result.get("text_eval")
        if isinstance(text_eval, dict):
            text_status = text_eval.get("status")
        image_eval = existing_result.get("image_eval")
        if isinstance(image_eval, list):
            for item in image_eval:
                if not isinstance(item, dict):
                    continue
                image_has_any = True
                if item.get("status") == "error":
                    image_has_error = True

    text_needs = text_status is None or text_status == "error"
    image_needs = (not image_has_any) or image_has_error
    force_full_rerun = False
    text_score = None

    if category == "reasoning" and subcategory in {"mmlu/mmmu", "puzzle"}:
        if text_status != "ok":
            force_full_rerun = True
        else:
            text_score, _ = extract_existing_text_score(existing_result)
            if text_score is None:
                force_full_rerun = True
            else:
                text_needs = False
                image_needs = (not image_has_any) or image_has_error

    if force_full_rerun:
        text_needs = True
        image_needs = True

    return {
        "needs_text": text_needs,
        "needs_image": image_needs,
        "force_full_rerun": force_full_rerun,
        "text_score": text_score,
        "has_any": text_needs or image_needs,
    }


def evaluate_reasoning_mmlu(
    model_dir,
    prompt_info,
    question_image,
    reference_image,
    prompt_dir,
    api_key,
    model_name,
    error_log_path,
    text_session,
    image_session,
    prompt_config,
    existing_result=None,
    skip_text=False,
    skip_image=False,
    existing_text_score=None,
):
    reference_text = prompt_info.get("reference_text")
    existing_result = existing_result if isinstance(existing_result, dict) else {}
    result = {
        "category": prompt_info.get("category"),
        "model_dir": str(model_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text_eval": existing_result.get("text_eval"),
        "image_eval": existing_result.get("image_eval") if isinstance(existing_result.get("image_eval"), list) else [],
    }

    if not question_image:
        raise ValueError("missing question_image for reasoning mmlu/mmmu")
    if not reference_text:
        raise ValueError("missing correct_option for reasoning mmlu/mmmu")

    generated_text = load_generated_text(model_dir)
    text_score = None
    if not skip_text:
        if generated_text is not None:
            try:
                prompt_path = Path(prompt_dir) / prompt_config["text"]
                prompt_data = load_yaml_prompt(prompt_path)
                template_vars = {
                    "question_image": "see attached question image",
                    "correct_option": reference_text,
                    "generated_answer": generated_text,
                }
                messages = build_messages(
                    prompt_data["system_prompt"],
                    prompt_data["template"],
                    template_vars,
                    image_paths=[question_image],
                )
                raw_response = call_text_model(text_session, api_key, model_name, messages)
                parsed = parse_json_maybe(raw_response)
                text_score = extract_score_value(parsed)
                result["text_eval"] = {
                    "status": "ok",
                    "input": {
                        "question_image": str(question_image),
                        "correct_option": reference_text,
                        "generated_text": generated_text,
                    },
                    "raw_response": raw_response,
                    "parsed": parsed,
                }
            except Exception as exc:
                if isinstance(exc, ApiResponseError):
                    log_error_with_raw(error_log_path, model_dir, f"text_eval_error: {exc}", exc.raw_response)
                else:
                    log_error(error_log_path, model_dir, f"text_eval_error: {exc}")
                error_payload = {"status": "error", "error": str(exc)}
                if isinstance(exc, ApiResponseError):
                    error_payload["raw_error_response"] = exc.raw_response
                result["text_eval"] = error_payload
        else:
            result["text_eval"] = {"status": "skipped", "reason": "text.json not found"}
    else:
        text_score = existing_text_score

    generated_images = list_generated_images(model_dir)
    if skip_image:
        return result
    if not generated_images:
        result["image_eval"] = [{"status": "skipped", "reason": "no images found"}]
        return result

    use_choice_image = generated_text is None or is_positive_score(text_score)
    eval_type = "choice_image" if use_choice_image else "image_text_balance"
    existing_map = build_existing_image_eval_map(result["image_eval"])
    result["image_eval"] = []
    for image_path in generated_images:
        existing_item = existing_map.get((str(image_path), eval_type))
        if should_keep_image_eval(existing_item):
            result["image_eval"].append(existing_item)
            continue
        try:
            if use_choice_image:
                prompt_path = Path(prompt_dir) / prompt_config["image"]
                prompt_data = load_yaml_prompt(prompt_path)
                template_vars = {
                    "question_image": "see attached question image",
                    "correct_option": reference_text,
                    "answer_image": "see attached answer image",
                }
                if "reference_answer_image" in (prompt_data.get("jinja_args") or []):
                    if not reference_image:
                        raise ValueError("missing reference_answer_image for reasoning choice image eval")
                    template_vars["reference_answer_image"] = "see attached reference image"
                    image_paths = [question_image, reference_image, image_path]
                else:
                    image_paths = [question_image, image_path]
            else:
                if generated_text is None:
                    raise ValueError("missing generated_text for balance evaluation")
                eval_type = "image_text_balance"
                prompt_path = Path(prompt_dir) / prompt_config["balance"]
                prompt_data = load_yaml_prompt(prompt_path)
                template_vars = {
                    "question_image": "see attached question image",
                    "generated_text": generated_text,
                    "answer_image": "see attached answer image",
                }
                image_paths = [question_image, image_path]

            messages = build_messages(
                prompt_data["system_prompt"],
                prompt_data["template"],
                template_vars,
                image_paths=image_paths,
            )
            raw_response = call_image_model(image_session, api_key, model_name, messages)
            parsed = parse_json_maybe(raw_response)
            result["image_eval"].append(
                {
                    "status": "ok",
                    "eval_type": eval_type,
                    "generated_image": str(image_path),
                    "question_image": str(question_image),
                    "correct_option": reference_text if use_choice_image else None,
                    "generated_text": generated_text if not use_choice_image else None,
                    "raw_response": raw_response,
                    "parsed": parsed,
                }
            )
        except Exception as exc:
            if isinstance(exc, ApiResponseError):
                log_error_with_raw(error_log_path, model_dir, f"image_eval_error: {exc}", exc.raw_response)
            else:
                log_error(error_log_path, model_dir, f"image_eval_error: {exc}")
            error_payload = {
                "status": "error",
                "eval_type": eval_type,
                "generated_image": str(image_path),
                "error": str(exc),
            }
            if isinstance(exc, ApiResponseError):
                error_payload["raw_error_response"] = exc.raw_response
            result["image_eval"].append(error_payload)

    return result


def evaluate_model_dir(
    model_dir,
    prompt_info,
    reference_image,
    problem_image,
    prompt_dir,
    api_key,
    model_name,
    error_log_path,
    text_session,
    image_session,
    existing_result=None,
    skip_text=False,
    skip_image=False,
    existing_text_score=None,
):
    category = prompt_info.get("category")
    reference_text = prompt_info.get("reference_text")
    problem_text = prompt_info.get("problem_text")
    gen_prompt = prompt_info.get("gen_prompt")
    existing_result = existing_result if isinstance(existing_result, dict) else {}
    result = {
        "category": category,
        "model_dir": str(model_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text_eval": existing_result.get("text_eval"),
        "image_eval": existing_result.get("image_eval") if isinstance(existing_result.get("image_eval"), list) else [],
    }

    subcategory = prompt_info.get("subcategory")
    prompt_config = get_prompt_config(category, subcategory)
    if not prompt_config:
        raise ValueError(f"Unsupported category: {category}")
    if category == "reasoning" and subcategory in {"mmlu/mmmu", "puzzle"}:
        return evaluate_reasoning_mmlu(
            model_dir=model_dir,
            prompt_info=prompt_info,
            question_image=problem_image,
            reference_image=reference_image,
            prompt_dir=prompt_dir,
            api_key=api_key,
            model_name=model_name,
            error_log_path=error_log_path,
            text_session=text_session,
            image_session=image_session,
            prompt_config=prompt_config,
            existing_result=result,
            skip_text=skip_text,
            skip_image=skip_image,
            existing_text_score=existing_text_score,
        )

    if prompt_config.get("text") and not skip_text:
        generated_text = load_generated_text(model_dir)
        if generated_text is not None:
            try:
                prompt_path = Path(prompt_dir) / prompt_config["text"]
                prompt_data = load_yaml_prompt(prompt_path)
                if category == "Multi Hop":
                    reference_label = "see attached reference image" if reference_image else "no reference image"
                    template_vars = {
                        "reference_image": reference_label,
                        "reference_text": reference_text,
                        "generated_text": generated_text,
                    }
                    image_paths = [reference_image] if reference_image else []
                elif category == "physics":
                    if not problem_text:
                        raise ValueError("missing problem_text for physics text eval")
                    template_vars = {
                        "problem": problem_text,
                        "reference_answer": reference_text,
                        "generated_text": generated_text,
                    }
                    image_paths = []
                elif category == "counting":
                    template_vars = {
                        "task_reference": reference_text,
                        "generated_text": generated_text,
                    }
                    image_paths = []
                elif category == "rule_based":
                    if not problem_text:
                        raise ValueError("missing und_prompt for rule_based text eval")
                    template_vars = {
                        "rule": problem_text,
                        "reference_text": reference_text,
                        "generated_text": generated_text,
                    }
                    image_paths = []
                elif category == "reasoning" and subcategory == "draw":
                    if not problem_image:
                        raise ValueError("missing question_image for reasoning draw text eval")
                    if not reference_image:
                        raise ValueError("missing reference_answer_image for reasoning draw text eval")
                    if not problem_text:
                        raise ValueError("missing task_text for reasoning draw text eval")
                    template_vars = {
                        "question_image": "see attached question image",
                        "task_text": problem_text,
                        "reference_answer_image": "see attached reference image",
                        "generated_text": generated_text,
                    }
                    image_paths = [problem_image, reference_image]
                else:
                    template_vars = {"generated_text": generated_text}
                    image_paths = []
                messages = build_messages(
                    prompt_data["system_prompt"],
                    prompt_data["template"],
                    template_vars,
                    image_paths=image_paths,
                )
                raw_response = call_text_model(text_session, api_key, model_name, messages)
                parsed = parse_json_maybe(raw_response)
                result["text_eval"] = {
                    "status": "ok",
                    "input": {
                        "reference_image": str(reference_image) if reference_image else None,
                        "problem_image": str(problem_image) if problem_image else None,
                        "reference_text": reference_text,
                        "problem_text": problem_text,
                        "generated_text": generated_text,
                    },
                    "raw_response": raw_response,
                    "parsed": parsed,
                }
            except Exception as exc:
                if isinstance(exc, ApiResponseError):
                    log_error_with_raw(error_log_path, model_dir, f"text_eval_error: {exc}", exc.raw_response)
                else:
                    log_error(error_log_path, model_dir, f"text_eval_error: {exc}")
                error_payload = {"status": "error", "error": str(exc)}
                if isinstance(exc, ApiResponseError):
                    error_payload["raw_error_response"] = exc.raw_response
                result["text_eval"] = error_payload
        else:
            result["text_eval"] = {"status": "skipped", "reason": "text.json not found"}

    if prompt_config.get("image") and skip_image:
        return result

    if prompt_config.get("image") and not skip_image:
        generated_images = list_generated_images(model_dir)
        if not generated_images:
            result["image_eval"] = [{"status": "skipped", "reason": "no images found"}]
        else:
            prompt_path = Path(prompt_dir) / prompt_config["image"]
            prompt_data = load_yaml_prompt(prompt_path)
            existing_map = build_existing_image_eval_map(result["image_eval"])
            result["image_eval"] = []
            for image_path in generated_images:
                try:
                    existing_item = existing_map.get((str(image_path), None))
                    if should_keep_image_eval(existing_item):
                        result["image_eval"].append(existing_item)
                        continue
                    if category == "Multi Hop":
                        reference_label = "see attached reference image" if reference_image else "no reference image"
                        generated_label = "see attached generated image"
                        template_vars = {
                            "reference_image": reference_label,
                            "reference_caption": reference_text,
                            "generated_image": generated_label,
                        }
                        image_paths = [reference_image, image_path]
                    elif category == "physics":
                        if not problem_text:
                            raise ValueError("missing problem_text for physics image eval")
                        template_vars = {
                            "problem_text": problem_text,
                            "reference_text": reference_text,
                        }
                        image_paths = [problem_image, reference_image, image_path]
                    elif category == "counting":
                        if not reference_image:
                            raise ValueError("missing reference_answer_image for counting image eval")
                        template_vars = {
                            "task_reference": reference_text,
                            "generated_image": "see attached generated image",
                            "reference_answer_image": "see attached reference image",
                        }
                        image_paths = [image_path, reference_image]
                    elif category == "rule_based":
                        if not gen_prompt:
                            raise ValueError("missing gen_prompt for rule_based image eval")
                        if not problem_image:
                            raise ValueError("missing original_image for rule_based image eval")
                        if not reference_image:
                            raise ValueError("missing reference_image for rule_based image eval")
                        template_vars = {
                            "original_image": "see attached original image",
                            "rule": gen_prompt,
                            "reference_image": "see attached reference image",
                            "generated_image": "see attached generated image",
                        }
                        image_paths = [problem_image, reference_image, image_path]
                    elif category == "reasoning" and subcategory == "draw":
                        if not problem_image:
                            raise ValueError("missing question_image for reasoning draw image eval")
                        if not reference_image:
                            raise ValueError("missing reference_answer_image for reasoning draw image eval")
                        if not problem_text:
                            raise ValueError("missing task_text for reasoning draw image eval")
                        template_vars = {
                            "question_image": "see attached question image",
                            "task_text": problem_text,
                            "reference_answer_image": "see attached reference image",
                            "answer_image": "see attached answer image",
                        }
                        image_paths = [problem_image, reference_image, image_path]
                    else:
                        template_vars = {}
                        image_paths = [image_path]
                    messages = build_messages(
                        prompt_data["system_prompt"],
                        prompt_data["template"],
                        template_vars,
                        image_paths=image_paths,
                    )
                    raw_response = call_image_model(image_session, api_key, model_name, messages)
                    parsed = parse_json_maybe(raw_response)
                    result["image_eval"].append(
                        {
                            "status": "ok",
                            "generated_image": str(image_path),
                            "reference_image": str(reference_image) if reference_image else None,
                            "problem_image": str(problem_image) if problem_image else None,
                            "reference_text": reference_text,
                            "problem_text": problem_text,
                            "raw_response": raw_response,
                            "parsed": parsed,
                        }
                    )
                except Exception as exc:
                    if isinstance(exc, ApiResponseError):
                        log_error_with_raw(error_log_path, model_dir, f"image_eval_error: {exc}", exc.raw_response)
                    else:
                        log_error(error_log_path, model_dir, f"image_eval_error: {exc}")
                    error_payload = {
                        "status": "error",
                        "generated_image": str(image_path),
                        "error": str(exc),
                    }
                    if isinstance(exc, ApiResponseError):
                        error_payload["raw_error_response"] = exc.raw_response
                    result["image_eval"].append(error_payload)

    return result


def write_result(model_dir, result):
    output_path = Path(model_dir) / "new_res.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


def evaluate_data_point(
    data_point_dir,
    prompt_dir,
    api_key,
    model_name,
    error_log_path,
    text_session,
    image_session,
    model_dirs=None,
    progress=None,
    simple_progress_counter=None,
):
    prompt_info = read_prompt_json(data_point_dir)
    category = prompt_info.get("category")
    subcategory = prompt_info.get("subcategory")
    reference_image = None
    problem_image = None
    if category == "Multi Hop":
        reference_image = find_reference_image(data_point_dir)
    elif category == "physics":
        problem_image = find_problem_image(data_point_dir)
        if not problem_image:
            problem_image = resolve_prompt_image(data_point_dir, prompt_info.get("problem_image_value"))
        reference_image = find_reference_image(data_point_dir)
        if not reference_image:
            reference_image = resolve_prompt_image(data_point_dir, prompt_info.get("expect_image_value"))
    elif category == "counting":
        reference_image = find_reference_image(data_point_dir)
    elif category == "rule_based":
        problem_image = find_problem_image(data_point_dir)
        reference_image = find_ref_only_image(data_point_dir)
    elif category == "reasoning":
        problem_image = find_problem_image(data_point_dir)
        reference_image = find_ref_only_image(data_point_dir)
    else:
        reference_image = find_reference_image(data_point_dir)
    if model_dirs is None:
        model_dirs = list_model_dirs(data_point_dir)
    total_models = len(model_dirs)
    for model_dir in model_dirs:
        if should_skip_model_dir(model_dir):
            if progress is not None:
                progress.update(1)
            elif simple_progress_counter is not None:
                simple_progress_counter["count"] += 1
                print(f"  [Models {simple_progress_counter['count']}/{total_models}] {model_dir.name} (skipped)")
            continue
        text_status, image_has_error = load_existing_status(model_dir)
        skip_text = text_status is not None and text_status != "error"
        skip_image = image_has_error is False and image_has_error is not None
        try:
            result = evaluate_model_dir(
                model_dir=model_dir,
                prompt_info=prompt_info,
                reference_image=reference_image,
                problem_image=problem_image,
                prompt_dir=prompt_dir,
                api_key=api_key,
                model_name=model_name,
                error_log_path=error_log_path,
                text_session=text_session,
                image_session=image_session,
                skip_text=skip_text,
                skip_image=skip_image,
            )
            write_result(model_dir, result)
        except Exception as exc:
            log_error(error_log_path, model_dir, f"model_dir_error: {exc}")
        if progress is not None:
            progress.update(1)
        elif simple_progress_counter is not None:
            simple_progress_counter["count"] += 1
            print(f"  [Models {simple_progress_counter['count']}/{total_models}] {model_dir.name}")


def resolve_images_for_data_point(data_point_dir, prompt_info):
    category = prompt_info.get("category")
    reference_image = None
    problem_image = None
    if category == "Multi Hop":
        reference_image = find_reference_image(data_point_dir)
    elif category == "physics":
        problem_image = find_problem_image(data_point_dir)
        if not problem_image:
            problem_image = resolve_prompt_image(data_point_dir, prompt_info.get("problem_image_value"))
        reference_image = find_reference_image(data_point_dir)
        if not reference_image:
            reference_image = resolve_prompt_image(data_point_dir, prompt_info.get("expect_image_value"))
    elif category == "counting":
        reference_image = find_reference_image(data_point_dir)
    elif category == "rule_based":
        problem_image = find_problem_image(data_point_dir)
        reference_image = find_ref_only_image(data_point_dir)
    elif category == "reasoning":
        problem_image = find_problem_image(data_point_dir)
        reference_image = find_ref_only_image(data_point_dir)
    else:
        reference_image = find_reference_image(data_point_dir)
    return reference_image, problem_image


def collect_pending_tasks(data_points, mod_value=None, error_log_override=None):
    tasks = []
    override_path = str(Path(error_log_override)) if error_log_override else None
    for data_point in data_points:
        prompt_info = read_prompt_json(data_point)
        reference_image, problem_image = resolve_images_for_data_point(data_point, prompt_info)
        model_dirs = list_model_dirs(data_point)
        for model_dir in model_dirs:
            existing_result = load_existing_result(model_dir)
            plan = plan_model_work(existing_result, prompt_info)
            if not plan["has_any"]:
                continue
            if override_path:
                error_log_path = override_path
            else:
                error_log_path = (
                    get_error_log_path_for_mod(mod_value)
                    if mod_value is not None
                    else get_error_log_path_for_mod(get_mod_group(data_point))
                )
            task = {
                "data_point_dir": data_point,
                "model_dir": model_dir,
                "prompt_info": prompt_info,
                "reference_image": reference_image,
                "problem_image": problem_image,
                "needs_text": plan["needs_text"],
                "needs_image": plan["needs_image"],
                "existing_text_score": plan["text_score"],
                "existing_result": None if plan["force_full_rerun"] else existing_result,
                "error_log_path": error_log_path,
            }
            tasks.append(task)
    return tasks


def run_task_pool(tasks, prompt_dir, api_key, model_name, text_session, image_session, progress=None):
    total_tasks = len(tasks)
    for index, task in enumerate(tasks, start=1):
        model_dir = task["model_dir"]
        try:
            result = evaluate_model_dir(
                model_dir=model_dir,
                prompt_info=task["prompt_info"],
                reference_image=task["reference_image"],
                problem_image=task["problem_image"],
                prompt_dir=prompt_dir,
                api_key=api_key,
                model_name=model_name,
                error_log_path=task["error_log_path"],
                text_session=text_session,
                image_session=image_session,
                existing_result=task["existing_result"],
                skip_text=not task["needs_text"],
                skip_image=not task["needs_image"],
                existing_text_score=task["existing_text_score"],
            )
            write_result(model_dir, result)
        except Exception as exc:
            log_error(task["error_log_path"], model_dir, f"model_dir_error: {exc}")
        if progress is not None:
            progress.update(1)
        else:
            print(f"[Tasks {index}/{total_tasks}] {model_dir.name}")


def confirm_auto():
    print("Auto mode will evaluate all data points under /home/cdp/cl/data.")
    answer = input("Type YES to continue: ").strip()
    return answer == "YES"


def prompt_for_data_point():
    value = input("Enter data point path (e.g., /home/cdp/cl/data/data_855): ").strip()
    return value


def prompt_for_mod_group():
    print("Select remainder group for data_id % 3.", flush=True)
    while True:
        value = input("Select remainder group to run (0/1/2): ").strip()
        if value in {"0", "1", "2"}:
            return int(value)
        print("Invalid input. Please enter 0, 1, or 2.")


def prompt_for_mod5_group():
    print("Select remainder group for data_id % 5.", flush=True)
    while True:
        value = input("Select remainder group to run (0/1/2/3/4): ").strip()
        if value in {"0", "1", "2", "3", "4"}:
            return int(value)
        print("Invalid input. Please enter 0, 1, 2, 3, or 4.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="run gpt5pro under /home/cdp/cl/data")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--prompt-dir", default=PROMPT_DIR_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--use-proxy", action="store_true", help="use proxy env vars instead of direct connect")
    parser.add_argument("--mod", type=int, choices=[0, 1, 2, 3, 4], help="run data points where id % 5 equals this value")
    parser.add_argument("--error-log", help="override error log path for all tasks")
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", API_KEY_DEFAULT))
    args = parser.parse_args()

    if not args.use_proxy:
        clear_proxy_env()
    text_session = create_requests_session(args.use_proxy)
    image_session = create_requests_session(args.use_proxy)
    if args.auto:
        if not confirm_auto():
            print("Auto mode cancelled.")
            return 0
        mod_value = args.mod if args.mod is not None else prompt_for_mod5_group()
        print(f"Selected remainder group (mod 5): {mod_value}", flush=True)
        data_root = Path(args.data_root)
        data_points = sorted(
            [p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("data_")]
        )
        filtered_points = []
        for data_point in data_points:
            data_id = parse_data_id(data_point)
            group = None if data_id is None else data_id % 5
            if group is None:
                continue
            if group == mod_value:
                filtered_points.append(data_point)
        data_points = filtered_points
        tasks = collect_pending_tasks(
            data_points,
            mod_value=None,
            error_log_override=args.error_log,
        )
        tasks = [t for t in tasks if Path(t["model_dir"]).name == "emu"]
        if not tasks:
            print("No pending tasks found.")
            return 0
        print(f"Pending tasks: {len(tasks)}", flush=True)
        if tqdm:
            with tqdm(total=len(tasks), desc="Pending tasks", unit="task") as task_bar:
                run_task_pool(
                    tasks=tasks,
                    prompt_dir=args.prompt_dir,
                    api_key=args.api_key,
                    model_name=args.model,
                    text_session=text_session,
                    image_session=image_session,
                    progress=task_bar,
                )
        else:
            run_task_pool(
                tasks=tasks,
                prompt_dir=args.prompt_dir,
                api_key=args.api_key,
                model_name=args.model,
                text_session=text_session,
                image_session=image_session,
                progress=None,
            )
    else:
        data_point = prompt_for_data_point()
        if not data_point:
            print("No data point provided.")
            return 1
        data_point_path = Path(data_point)
        if not data_point_path.exists():
            print("Data point path not found.")
            return 1
        tasks = collect_pending_tasks(
            [data_point_path],
            mod_value=None,
            error_log_override=args.error_log,
        )
        tasks = [t for t in tasks if Path(t["model_dir"]).name == "emu"]
        if not tasks:
            print("No pending tasks found.")
            return 0
        if tqdm:
            with tqdm(total=len(tasks), desc=data_point_path.name, unit="task") as task_bar:
                run_task_pool(
                    tasks=tasks,
                    prompt_dir=args.prompt_dir,
                    api_key=args.api_key,
                    model_name=args.model,
                    text_session=text_session,
                    image_session=image_session,
                    progress=task_bar,
                )
        else:
            run_task_pool(
                tasks=tasks,
                prompt_dir=args.prompt_dir,
                api_key=args.api_key,
                model_name=args.model,
                text_session=text_session,
                image_session=image_session,
                progress=None,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
