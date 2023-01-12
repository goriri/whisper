import logging
import json
import torch

from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import pipeline

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

model_path = '/opt/ml/model'
logger.info("Libraries are loaded")


def model_fn(model_dir):
    device = get_device()

    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
    # model = pipeline(task='automatic-speech-recognition', model=model_dir, device=device)
    logger.info("Model is loaded")

    return model


def input_fn(json_request_data, content_type='application/json'):
    input_data = json.loads(json_request_data)
    logger.info("Input data is processed")

    return input_data


def predict_fn(input_data, model):
    logger.info("Starting inference.")
    device = get_device()

    logger.info(input_data)

    speech_array = input_data['speech_array']
    sampling_rate = input_data['sampling_rate']

    processor = WhisperProcessor.from_pretrained(model_path)
    input_features = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcript = processor.batch_decode(predicted_ids, skip_special_tokens = True)

    # transcript = model(audio)["text"]

    return transcript


def output_fn(transcript, accept='application/json'):
    return json.dumps(transcript), accept


def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device

