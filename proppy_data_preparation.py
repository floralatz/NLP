
import pandas as pd
import pickle

###### DATA PREP ######

# Delete columns
df = pd.read_csv("proppy_data/proppy_1.0.dev.tsv",sep='\t')
df = df.drop(df.columns[1:14], axis=1)

# Name columns
df.columns.values[0] = "text"
df.columns.values[1] = "labels"
print(df.head())
print(len(df.index))


# Make dataset out of the first 5 and last 5 rows
first_rows = df[0:5]
last_rows = df[5118:5123]

smaller_df = pd.concat([first_rows, last_rows]).reset_index(drop=True)
print(smaller_df)

# Change labels 1 = propaganda 0 = no propaganda
# so far: “propagandistic” (positive class) or “non-propagandistic” (negative class)

smaller_df['labels'] = smaller_df['labels'].replace([-1], 0)
print(smaller_df)

# save as pickles
filename = "proppy_data_evaluation" + ".pickle"
with open(filename, 'wb') as f:
    pickle.dump(smaller_df, f)
