import gzip
import json
import os

import numpy
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_cve_data():
    all_vuls = []
    for i in os.listdir('./nvd'):
        if i == 'nvdcve-1.1-2023.json.gz':
            continue
        reader = json.load(gzip.open(f'./nvd/{i}'))
        vulns = reader['CVE_Items']
        parsed = []
        for data in vulns:
            try:
                # cvss vector start
                av = data['impact']['baseMetricV3']['cvssV3']['attackVector']
                # desc here
                desc = data['cve']['description']['description_data'][0]['value']

                cve_id = data['cve']['CVE_data_meta']['ID']

                parsed.append([cve_id, desc, av])
            except:
                continue
        print('load %s from file: %s'%(len(parsed), i))
        all_vuls.extend(parsed)
    return all_vuls


def set_label(df, column_name):
    label = preprocessing.LabelEncoder()
    if os.path.exists(column_name + '.npy'):
        label.classes_ = numpy.load(f'{column_name}.npy', allow_pickle=True)
        df[f'{column_name}_label'] = label.transform(df[column_name])
    else:
        df[f'{column_name}_label'] = label.fit_transform(df[column_name])
        numpy.save(column_name + '.npy', label.classes_)
    return df


def build_index():
    cves = load_cve_data()
    print(len(cves))
    df = pd.DataFrame(cves, columns=['cve_id', 'desc', 'av'])

    for columns in ['av']:
        df = set_label(df, columns)
    return df


def do_learn(df, label_column, base_model='roberta', sub_model='roberta-base'):
    from simpletransformers.classification import ClassificationModel, ClassificationArgs
    new_df = pd.DataFrame()
    new_df['text'] = df['desc']
    new_df['labels'] = df[label_column]
    train_df, test_df = train_test_split(new_df)
    train_df = pd.DataFrame(train_df)
    # train_df.columns = ["text", "labels"]
    model_args = ClassificationArgs(num_train_epochs=1, output_dir=f'./test_{label_column}_{base_model}_{sub_model}')
    model_args.use_multiprocessing = False
    model_args.early_stopping_delta = 0.05
    model_args.early_stopping_metric = "mcc"
    model_args.use_multiprocessing_for_evaluation = False
    model_args.use_multiprocessed_decoding = False

    model_args.train_batch_size = 128
    model_args.save_steps = -1
    model_args.save_model_every_epoch = False

    model = ClassificationModel(
        base_model, sub_model, args=model_args, num_labels=len(new_df['labels'].unique()), use_cuda=True
    )
    model.train_model(train_df)

    result = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)
    print(result[0])


if __name__ == '__main__':
    df = build_index()
    for label in ['av']:
        do_learn(df, f'{label}_label')
