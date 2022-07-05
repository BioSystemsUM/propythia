from .descriptors import DNADescriptor
import pandas as pd


def calculate_feature(data: pd.DataFrame):
    list_feature = []
    count = 0
    for seq in data['sequence']:
        res = {'sequence': seq}
        dna = DNADescriptor(seq)
        feature = dna.get_descriptors()
        res.update(feature)
        list_feature.append(res)

        # print progress every 100 sequences
        if count % 100 == 0:
            print(count, '/', len(data))

        count += 1
    print("Done!")
    df = pd.DataFrame(list_feature)
    return df


def process_lists(fps_x, field):
    l = fps_x[field].to_list()
    new_df = pd.DataFrame(l)
    new_df.columns = [str(field) + "_" + str(i) for i in new_df.columns]
    fps_x.drop(field, axis=1, inplace=True)
    return new_df


def process_lists_of_lists(fps_x, field):
    l = fps_x[field].to_list()
    new_df = pd.DataFrame(l)
    new_df.columns = [str(field) + "_" + str(i) for i in new_df.columns]
    empty_val = {} if field == "enhanced_nucleic_acid_composition" else []
    small_processed = []
    for f in new_df.columns:
        col = [empty_val if i is None else i for i in new_df[f].to_list()]
        sub = pd.DataFrame(col)
        sub.columns = [str(f) + "_" + str(i) for i in sub.columns]
        small_processed.append(sub)
    fps_x.drop(field, axis=1, inplace=True)
    return small_processed


def normalization(fps_x):
    lists = ["nucleic_acid_composition", "dinucleotide_composition", "trinucleotide_composition",
             "k_spaced_nucleic_acid_pairs", "kmer", "PseDNC", "PseKNC", "DAC", "DCC", "DACC", "TAC", "TCC", "TACC"]
    lists_of_lists = [
        "accumulated_nucleotide_frequency"
    ]

    small_processed = []
    for i in lists:
        new_df = process_lists(fps_x, i)
        small_processed.append(new_df)

    for i in lists_of_lists:
        smaller_processed = process_lists_of_lists(fps_x, i)
        small_processed += smaller_processed

    new_fps_x = pd.concat([fps_x, *small_processed], axis=1)
    return new_fps_x


def calculate_and_normalize(data: pd.DataFrame):
    features = calculate_feature(data)
    fps_y = data['label']
    fps_x = features.loc[:, features.columns != 'label']
    fps_x = fps_x.loc[:, fps_x.columns != 'sequence']
    fps_x = normalization(fps_x)
    return fps_x, fps_y
