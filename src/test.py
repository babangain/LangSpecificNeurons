import pandas as pd
import numpy as np
keys = ["en", "bn", "hi"]
lang_to_neuron = {
    "en": [(1,0),(2,0),(3,0),(4,0),(5,0)],
    "bn": [(2,0),(5,0),(6,0),(7,0),(8,0)],
    "hi": [(3,0),(6,0),(7,0),(9,0),(10,0)]
}
matrix = pd.DataFrame(np.zeros((len(keys), len(keys)), dtype=np.int64), index=keys, columns=keys)

# Step 1: Calculate neurons specific to each language
lang_specific_neurons = {}
for key in keys:
    lang_specific_neurons[key] = {tuple(i) for i in lang_to_neuron[key]}

# Step 2: Calculate exclusive overlaps
for i, key1 in enumerate(keys):
    for j, key2 in enumerate(keys):
        if i <= j:
            data1 = lang_specific_neurons[key1]
            data2 = lang_specific_neurons[key2]
            overlap = data1 & data2
            for other_key in set(keys)-set([key1, key2]):
                overlap -= lang_specific_neurons[other_key]
            
            count_common = len(overlap)
            matrix.at[key1, key2] = count_common
            matrix.at[key2, key1] = count_common

print(matrix)
