import pandas as pd
from typing import List
from .descriptors import DNADescriptor

def _calculate_descriptors(data: pd.DataFrame, descriptor_list: List) -> pd.DataFrame:
    """
    From a dataset of sequences and labels, this function calculates the descriptors and returns a dataframe with them.
    The user can also specify which descriptors to calculate.
    """
    list_feature = []
    count = 0
    for seq in data['sequence']:
        res = {'sequence': seq}
        dna = DNADescriptor(seq)
        features = dna.get_descriptors(descriptor_list)
        res.update(features)
        list_feature.append(res)

        # print progress every 100 sequences
        if count % 100 == 0:
            print(count, '/', len(data))

        count += 1
    print("Done!")
    df = pd.DataFrame(list_feature)
    return df


def _process_lists(fps_x, field):
    """
    A helper function to normalize lists.
    """
    l = fps_x[field].to_list()
    new_df = pd.DataFrame(l)
    new_df.columns = [str(field) + "_" + str(i) for i in new_df.columns]
    fps_x.drop(field, axis=1, inplace=True)
    return new_df


def _process_lists_of_lists(fps_x, field):
    """
    A helper function to normalize lists of lists.
    """
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



def normalization(fps_x, descriptor_list):
    """
    Because the model cannot process data in dictionaries and lists, the descriptors that produce these forms must still be normalized.

    To normalize the data, dicts and lists need to "explode" into more columns. 

    E.g. dicts:

    | descriptor_hello |
    | ---------------- |
    | {'a': 1, 'b': 2} |

    will be transformed into:

    | descriptor_hello_a | descriptor_hello_b |
    | ------------------ | ------------------ |
    | 1                  | 2                  |

    E.g. lists:

    | descriptor_hello |
    | ---------------- |
    | [1, 2, 3]        |

    will be transformed into:

    | descriptor_hello_0 | descriptor_hello_1 | descriptor_hello_2 |
    | ------------------ | ------------------ | ------------------ |
    | 1                  | 2                  | 3                  |
    """
    lists = ["nucleic_acid_composition", "dinucleotide_composition", "trinucleotide_composition",
             "k_spaced_nucleic_acid_pairs", "kmer", "PseDNC", "PseKNC", "DAC", "DCC", "DACC", "TAC", "TCC", "TACC"]
    lists_of_lists = [
        "accumulated_nucleotide_frequency"
    ]
    
    # update to be normalized lists with only columns the user wants
    if(descriptor_list != []):
        lists = [l for l in lists if l in descriptor_list]
        lists_of_lists = [l for l in lists_of_lists if l in descriptor_list]

    small_processed = []
    for i in lists:
        new_df = _process_lists(fps_x, i)
        small_processed.append(new_df)

    for i in lists_of_lists:
        smaller_processed = _process_lists_of_lists(fps_x, i)
        small_processed += smaller_processed

    new_fps_x = pd.concat([fps_x, *small_processed], axis=1)
    return new_fps_x


def calculate_and_normalize(data: pd.DataFrame, descriptor_list: list = []) -> pd.DataFrame:
    """
    This function calculates the descriptors and normalizes the data all at once from a dataframe of sequences and labels. The user can also specify which descriptors to calculate.
    """
    features = _calculate_descriptors(data, descriptor_list)
    if 'label' in data:
        fps_y = data['label']
    else:
        fps_y = None
    fps_x = features.loc[:, features.columns != 'label']
    fps_x = fps_x.loc[:, fps_x.columns != 'sequence']
    fps_x = normalization(fps_x, descriptor_list)
    return fps_x, fps_y

if __name__ == "__main__":
    from sequence import ReadDNA
    reader = ReadDNA()
    filename = 'datasets/primer/dataset.csv'
    data = reader.read_csv(filename=filename, with_labels=True)
    fps_x, fps_y = calculate_and_normalize(data)
    print(fps_x)