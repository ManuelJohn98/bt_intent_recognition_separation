# generatestatistics.py

import re

FILES = [
    "Halleneroeffnung_A-DRZ_1", 
    "Halleneroeffnung_A-DRZ_2", 
    "Halleneroeffnung_A-DRZ_3",
    "Halleneroeffnung_Durchgang_1", 
    "Halleneroeffnung_Durchgang_2", 
    "Halleneroeffnung_Durchgang_3",
    "Halleneroeffnung_Durchgang_4",
    "Viersen_test1_1", 
    "Viersen_test2_1",
    "Viersen_test3"
    ]

label_dict = {'multi_intents': 0}

def statistics(filename):
    new_paragraph = True
    current_label = ''

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '\n':
                new_paragraph = True
                continue
            if line.startswith('#'):
                continue
            elems = line.split(sep='\t')
            label = re.match(r'[a-z_\\\/]+', elems[3]).group(0)
            if label == '_':
                continue
            if new_paragraph:
                if label in label_dict:
                    label_dict[label] += 1
                else:
                    label_dict[label] = 1
                new_paragraph = False
                current_label = label
            if current_label != label:
                if label in label_dict:
                    label_dict[label] += 1
                else:
                    label_dict[label] = 1
                current_label = label
                label_dict['multi_intents'] += 1
    return label_dict

for file in FILES:
    file = file + '.tsv'
    statistics(file)


with open("statistics.txt", 'w', encoding='utf-8') as f:
    # f.write(f"# {filename}\n")
    for label in label_dict:
        f.write(f"{label} - {label_dict[label]}\n")
    f.write('\n')
