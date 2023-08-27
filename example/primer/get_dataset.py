import requests
import pandas as pd

# from https://colab.research.google.com/drive/17E4h5aAOioh5DiTo7MZg4hpL6Z_0FyWr#scrollTo=IA9FJeQkr1Ze
SEQUENCES_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/sequences.txt'

sequences = requests.get(SEQUENCES_URL).text.split('\n')
sequences = list(filter(None, sequences))  # This removes empty sequences.


LABELS_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/labels.txt'

labels = requests.get(LABELS_URL).text.split('\n')
labels = list(filter(None, labels))  # removes empty sequences


# Let's print the first few sequences.
df = pd.DataFrame(sequences, index=np.arange(1, len(sequences)+1),
             columns=['sequence']) #.head()
df['label'] = labels
df.to_csv('primer/dataset.csv')