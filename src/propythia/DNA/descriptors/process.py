from sequence import ReadDNA
import csv


def write_dict_to_csv(d: dict, filename: str):
    """
    Writes a dictionary to a csv file.
    """
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for category, sequences in d.items():
            for key, val in sequences.items():
                writer.writerow([category, key, val])


if __name__ == '__main__':
    dna = ReadDNA()
    dna.read_fasta_in_folder('enhancer_dataset')
    write_dict_to_csv(dna.d, 'enhancer_dataset.csv')
