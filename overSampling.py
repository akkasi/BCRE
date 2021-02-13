from random import choices, shuffle
def RandomOversampling(df):
    df = df.reset_index(drop=True)
    labels = list(set(df.label.values))
    indexDict = {}
    for l in labels:
        index = df.index
        indexDict[l] = list(index[df.label == l])
    lenghts = [len(indexDict[l]) for l in indexDict]
    n_negatives = max(lenghts)
    oversampledDict = {}
    for l in indexDict:
        if len(indexDict[l]) != n_negatives:
            oversamples = choices(indexDict[l], k=n_negatives - len(indexDict[l]))
            oversampledDict[l] = indexDict[l] + oversamples
        else:
            oversampledDict[l] = indexDict[l]
    final_list = []
    for l in oversampledDict:
        final_list.extend(oversampledDict[l])
    shuffle(final_list)
    df = df.iloc[final_list, :]
    return df.reset_index(drop=True)