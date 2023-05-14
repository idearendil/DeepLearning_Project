from tqdm import tqdm
import pickle
from transformers import ElectraTokenizer

import os
import csv

MBTIs = ["ESTJ", "ESTP", "ESFJ", "ESFP",
         "ENTJ", "ENTP", "ENFJ", "ENFP",
         "ISTJ", "ISTP", "ISFJ", "ISFP",
         "INTJ", "INTP", "INFJ", "INFP"]


def emot_num(emo):
    return MBTIs.index(emo)


# Delete Current Processed Files
processed_dir = "dataset/processed"
for f in tqdm(os.listdir(processed_dir)):
    os.remove(os.path.join(processed_dir, f))


path_name = os.getcwd()
unprocessed_dir = "dataset/unprocessed"

unprocessed_csv_files = []

file_names_unprocessed = os.listdir(os.path.join(path_name, unprocessed_dir))
for file_name in file_names_unprocessed:
    if file_name.endswith(".csv"):
        unprocessed_csv_files.append(file_name)

tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")

for file_id, file_name in enumerate(tqdm(unprocessed_csv_files)):
    with open(os.path.join(path_name, unprocessed_dir, file_name), newline='') as file:

        reader = csv.reader(file)
        iterate = 0
        for row in reader:
            iterate += 1
            if iterate < 1:
                continue

            sample_point = {}

            sample_point["text"] = row[1]
            sample_point["MBTI"] = row[2]

            inputs = tokenizer(sample_point["text"],
                               return_tensors='pt',
                               truncation=True,
                               max_length=256,
                               pad_to_max_length=True,
                               add_special_tokens=True
                               )
            sample_point["input_ids"] = inputs['input_ids'][0]
            sample_point["attention_mask"] = inputs['attention_mask'][0]

            sample_name = "dataset/processed/" + str(file_id) + "_" + str(iterate-1)

            with open(sample_name+'.pickle', 'wb') as handle:
                pickle.dump(sample_point, handle, protocol=pickle.HIGHEST_PROTOCOL)
