from datasets import load_dataset
from pprint import pprint
import json


dataset = load_dataset("OpenScholar/OS_Train_Data")


pprint(dataset['train'][1])

with open("text_sample2.json", "w") as f:
    f.write(json.dumps(dataset["train"][1]))