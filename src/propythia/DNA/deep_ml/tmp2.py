import pandas as pd
from os.path import exists
import pickle
import sys
sys.path.append('../../../../src/')
from propythia.DNA.descriptors.descriptors import DNADescriptor


def calculate_feature(data):
    specifics = []
    list_feature = []
    count = 0
    not_valid = 0
    for seq in data['sequence']:
        res = {'sequence': seq}
        dna = DNADescriptor(seq)
        
        feature = dna.get_descriptors(specifics=specifics)
        res.update(feature)
        list_feature.append(res)
        
        # print progress every 100 sequences
        if count % 100 == 0:
            print(count, '/', len(data))

        count += 1
    print("Done!")
    df = pd.DataFrame(list_feature)
    return df, not_valid


def calculate(filename):    
    if exists("datasets/head_features.pkl"):
        with open("datasets/head_features.pkl", "rb") as f:
            features = pickle.load(f)
        print("features were loaded!")
    else:
        dataset = pd.read_csv(filename)
        features, not_valid = calculate_feature(dataset)
        print("Not valid:", not_valid)
        with open("datasets/head_features.pkl", "wb") as f:
            pickle.dump(features, f)
    return features


# works for all need dict normalization
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

# ---------------------------------------------------------------------------------------------

no_need_normalization = ["length", "at_content", "gc_content"]
lists = ["nucleic_acid_composition","dinucleotide_composition","trinucleotide_composition","k_spaced_nucleic_acid_pairs","kmer","PseDNC", "PseKNC", "DAC", "DCC", "DACC", "TAC","TCC","TACC", "accumulated_nucleotide_frequency"]
lists_of_lists = ["enhanced_nucleic_acid_composition","nucleotide_chemical_property", "binary"]


def main():
    # features = calculate("datasets/essential_genes_features.csv")
    features = calculate("datasets/testing_head.csv")
    
    fps_x = features.loc[:, features.columns != 'label']
    fps_x = fps_x.loc[:, fps_x.columns != 'sequence']
    
    # count time 
    import time
    start = time.time()
    small_processed = []
    for i in lists:
        new_df = process_lists(fps_x, i)
        small_processed.append(new_df)
        
    for i in lists_of_lists:
        smaller_processed = process_lists_of_lists(fps_x, i)
        small_processed += smaller_processed
    
    end = time.time()
    print("Time:", end - start)
    
    # concat final with original
    fps_x = pd.concat([fps_x, *small_processed], axis=1)
    fps_x.to_csv("datasets/head_normalized3.csv", index=False)
    
if __name__ == "__main__":
    main()