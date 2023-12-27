#!/usr/bin/env python
# coding: utf-8

from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import json
import numpy as np
from PIL import Image
import random

class Blip2:
    modelName="Salesforce/blip2-opt-2.7b"
    model = Blip2ForConditionalGeneration.from_pretrained(modelName, torch_dtype=torch.float16) 

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def gettext(self, image, prompt="Describe the person in the photo in detail, including what he or she is wearing and what he or she is doing."):
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=40)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text