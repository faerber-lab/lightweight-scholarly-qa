import json
import os

pubmedqa = []
with open("/home/s9650707/s9650707-llm_secrets/datasets/scholarqabench/ScholarQABench/data/single_paper_tasks/pubmed_test.jsonl", 'r') as f:
    for line in f:
        pubmedqa.append(json.loads(line)['input'])



scholarqabench_dir = "/data/horse/ws/s9650707-llm_secrets/datasets/scholarqabench/ScholarQABench/data/scholarqa_multi/"
scholarqabench_input_file = os.path.join(scholarqabench_dir, "human_answers.json")
with open(scholarqabench_input_file, "r") as f:
    scholarqabench_data = json.load(f)

scholarqabench = [_['input'] for _ in scholarqabench_data]


with open("qa_questions.json", "w") as f:
    json.dump({'pubmedqa': pubmedqa, 'scholarqabench_multiqa': scholarqabench}, f)