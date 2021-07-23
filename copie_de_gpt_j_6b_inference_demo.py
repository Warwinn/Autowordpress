import os
import time
import requests 
import json

import numpy as np

import jax
from jax.config import config
from jax.experimental import maps

import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

import people_also_ask as paa
from fake_useragent import UserAgent

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

"""### TPU Settings
---
"""



colab_tpu_addr = os.environ['COLAB_TPU_ADDR'].split(':')[0]
url = f'http://{colab_tpu_addr}:8475/requestversion/tpu_driver0.1_dev20210607'
requests.post(url)

# The following is required to use TPU Driver as JAX's backend.
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://" + os.environ['COLAB_TPU_ADDR']

"""## Setup GPT-J-6B
---
"""

params = {
  "layers": 28,
  "d_model": 4096,
  "n_heads": 16,
  "n_vocab": 50400,
  "norm": "layernorm",
  "pe": "rotary",
  "pe_rotary_dims": 64,

  "seq": 2048,
  "cores_per_replica": 8,
  "per_replica_batch": 1,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]


params["sampler"] = nucleaus_sample

# here we "remove" the optimizer parameters from the model (as we don't need them for inference)
params["optimizer"] = optax.scale(0)

mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

total_batch = per_replica_batch * jax.device_count() // cores_per_replica

network = CausalTransformer(params)

network.state = read_ckpt(network.state, "step_383500/", devices.shape[1])

network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))

def infer(context, top_p=0.9, temp=1.0, gen_len=512):
    """ Generate text from a sentence"""
    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(tokenizer.decode(o))

    return samples

def get_related_keywords(keyword, nb_keywords):
    """Return a list containing the wanted number of keywords related to a given keyword"""
    keyword.replace(" ", "+")
    url = "http://suggestqueries.google.com/complete/search?output=chrome&hl=en&gl=en&q=" + keyword
    ua = UserAgent()
    headers = {"user-agent": ua.chrome}
    response = requests.get(url, headers=headers, verify=False)
    related_keywords = []
    suggestions = json.loads(response.text)
    for i in range(nb_keywords):
        try:
            word = suggestions[1][i]
            related_keywords.append(word)
        except IndexError:
            break
    return related_keywords

def generate_qs_as(keyword, nb_questions, nb_articles):
  """Return a dict with the row containing the keyword and a given number of questions/answers for this keyword"""
  output={}
  for question in paa.get_related_questions(keyword, nb_questions-1):
    question = question.split('Search')[0]
    articles = []
    start = time.time()
    print(question)
    print("Generating ...")
    for i in range(nb_articles):
      article = infer(question, gen_len=600 )[0]
      ps = article.split("\n\n")
      p_with_sub = []
      for p in ps:
        if len(p) > 300:
          subtitle = generate_subtitle(p)
          p_with_sub.append({
              "subtitle" : subtitle, 
              "paragraph" : p
          })
      if p_with_sub:
        articles.append(p_with_sub)
    if articles:
      output[question] = articles
    spent_time = round(time.time() - start)
    #print(f"completion done in {spent_time}s - question : {question}")
    print(f"completion done in {spent_time}s")
    print("#===================#")
  return output

def generate_subtitle(text):
    # declare toenizer and model
    tokenizer = AutoTokenizer.from_pretrained("snrspeaks/t5-one-line-summary")
    model = AutoModelForSeq2SeqLM.from_pretrained("snrspeaks/t5-one-line-summary")
    # Perform translation and decode the output
    tokenized_text = tokenizer(text, return_tensors="pt")
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]
    return translated_text

def generate_article (words, nb_keywords=5, nb_questions=2, nb_articles=1):
    if isinstance(words, str):
        words = [words]

    output = {}
    for word in words:
        output[word]={}
        keywordlist = get_related_keywords(word, nb_keywords)
        for keyword in  keywordlist:
            output[word][keyword] = generate_qs_as(keyword, nb_questions, nb_articles)
            
    return json.dumps(output, indent = 4)

words = ["amazonia", "hurricane", "hypnosis"]

output= generate_article(words, nb_keywords=4, nb_questions=3, nb_articles=2)

parsed = json.loads(output)

with open(f"{'_'.join(words)}.json", 'w') as outfile:
    json.dump(parsed, outfile)