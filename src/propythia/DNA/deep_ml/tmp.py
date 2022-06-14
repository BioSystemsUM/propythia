import pandas as pd
import ast
import sys
import json
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
    dataset = pd.read_csv(filename)
    features, not_valid = calculate_feature(dataset)
    print("Not valid:", not_valid)
    features.to_csv("datasets/head.csv", index=False)

# ---------------------------------------------------------------------------------------------

no_need_normalization = ["length", "at_content", "gc_content"]
need_dict_normalization = ["nucleic_acid_composition", "enhanced_nucleic_acid_composition","dinucleotide_composition","trinucleotide_composition","k_spaced_nucleic_acid_pairs","kmer","PseDNC", "PseKNC"]
need_list_normalization = ["nucleotide_chemical_property", "accumulated_nucleotide_frequency", "DAC", "DCC", "DACC", "TAC","TCC","TACC", "binary"]

def normalize_dict(d, field):
    df = pd.json_normalize(d)
    df.columns = [str(field) + "_" + str(i) for i in df.columns]
    for f in df.columns:
        if any(isinstance(elem, dict) for elem in df[f]):
            df = pd.concat([df, normalize_dict(df[f], f)], axis=1)
            df.drop(f, axis=1, inplace=True)
    return df

def normalize_list(l, field):
    df = pd.DataFrame([[] if i is None else i for i in l.to_list()])    
    df.columns = [str(field) + "_" + str(i) for i in df.columns]
    
    for f in df.columns:
        if isinstance(df[f][0], list):
            df = pd.concat([df, normalize_list(df[f], f)], axis=1)
            df.drop(f, axis=1, inplace=True)
    return df

def process(fps_x):
    new_fps_x = pd.DataFrame()
    count = 0
    for col in fps_x.columns:
        if col in need_dict_normalization:
            new_fps_x = pd.concat([new_fps_x, normalize_dict(fps_x[col], col)], axis=1)
        elif col in need_list_normalization:
            new_fps_x = pd.concat([new_fps_x, normalize_list(fps_x[col], col)], axis=1)
        else:
            new_fps_x[col] = fps_x[col].to_numpy()
        count += 1
        # print("col:", col, ":", count, '/', len(fps_x.columns))
    return new_fps_x

def main():
    # calculate("datasets/testing_head.csv")
    # features = pd.read_csv('datasets/head.csv')
    features = pd.read_csv('datasets/essential_genes_features.csv', nrows=5)
    for i in features.columns[4:]:
        features[i] = features[i].apply(lambda x: ast.literal_eval(x)) 
    
    fps_x = features.loc[:, features.columns != 'label']
    fps_x = fps_x.loc[:, fps_x.columns != 'sequence']
    
    fps_x = process(fps_x)
    fps_x.to_csv("datasets/head_normalized.csv", index=False)
    
if __name__ == "__main__":
    main()