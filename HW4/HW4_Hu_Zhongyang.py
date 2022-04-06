
from pymagnitude import *
vectors = Magnitude("GoogleNews-vectors-negative300.magnitude")



print('the dimensionality is', vectors.dim)


print(vectors.most_similar(vectors.query("picnic"), topn=6))
print(vectors.most_similar("picnic", topn=5))
print('the word is not like others:', vectors.doesnt_match(['tissue', 'papyrus','manila', 'newsprint', 'parchment', 'gazette']))

print('the most similar is',vectors.most_similar(positive = ["leg",'throw'], negative = ['jump']))