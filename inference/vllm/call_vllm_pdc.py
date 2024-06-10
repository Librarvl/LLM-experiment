import requests
import concurrent.futures
from typing import List, Tuple, Optional
import time
import os
import tqdm
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed



def sub_query(url, query, model_name):
    if query == "-1":
        return "-1", None

    session = requests.Session()
    json_data = {
            "model": model_name,
            "messages": [{"role": "system", "content": "You are a helpful assistant"},
                         {"role": "user", "content": query}],
            # "messages": [{"role": "user", "content": query}],
        }
    
    config_dict = {
        "n": 1,
        "temperature":1,
        "top_p":0.7,
        "top_k":5,
        # # "use_beam_search": True,
        # # "num_beams": 5,
        "length_penalty":1,
        "repetition_penalty": 1.05,
        # "min_tokens": 200,
        "max_tokens": 4000,
        "frequency_penalty":0.5
    }
    json_data.update(config_dict)

    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = session.post(url, json=json_data)
            response.raise_for_status()
            res_text = response.json()
            final_response = (
                res_text.get("choices", [{}])[0].get("message", {}).get("content")
            )
            # count += 1
            # print(count)
            print("has done")
            return final_response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt+1} failed, retrying...")
                print(len(json_data["messages"][1]['content']))
                continue
            else:
                print(f"Final attempt failed with error: {e}")
                return None

    return None


if __name__ == "__main__":

    ip_address, port = "10.96.202.82", 8000
    url = "http://%s:%d/v1/chat/completions" % (ip_address, port)
    # model_name = "./afs/Qwen1.5-72B/word2ppt_api2"
    model_name = "./models/Qwen1.5-72B/"

    result = sub_query(url, "你好", model_name)
    print(result)