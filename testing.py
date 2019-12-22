import sys
import numpy as np
import fileinput, string, sys, re
import math
from statistics import mean 

def preprocess(input_file):
    input_file = re.sub('[.,)(?"“”:;!]', r'', input_file, flags = re.M)
    input_file = re.sub("’s ", r' ', input_file, flags = re.M)
    input_file = re.sub("'s ", r' ', input_file, flags = re.M)
    input_file = re.sub("' s ", r' ', input_file, flags = re.M)
    input_file = re.sub(" – ", r' ', input_file, flags = re.M)
    input_file = re.sub("– ", r'', input_file, flags = re.M)
    input_file = re.sub("- ", r'', input_file, flags = re.M)
    input_file = re.sub(" -", r'', input_file, flags = re.M)
    input_file = re.sub("['’]", r'', input_file, flags = re.M)
    input_file = re.sub("[a-z-0-9]*/[a-z-0-9]*", r'', input_file, flags = re.M)
    input_file = re.sub("[A-Z]*[0-9]", r'', input_file, flags = re.M)
    input_file = re.sub("[0-9]", r'', input_file, flags = re.M)
    input_file = re.sub("  [ ]*", r' ', input_file, flags = re.M)
    input_file = re.sub("\n", r' ', input_file, flags = re.M)
    return input_file

def cosine_similarity(v1, v2):
    a1 = np.asarray(v1)
    a2 = np.asarray(v2)
    mag_a1 = np.dot(a1,a1)
    mag_a2 = np.dot(a2,a2)
    sim = np.dot(a1,a2)/(np.sqrt(mag_a1)*np.sqrt(mag_a2))
    return sim

def cos(s1,s2):
    l = set(s1).union(set(s2))
    d1 = {}
    d2 = {}

    for e in l:
        d1[e] = 0
        d2[e] = 0
    for w in s1:
        d1[w]+=1;
    for w in s2:
        d2[w]+=1;

    for w in d1.keys():
        if d1[w]==0:
            d1[w] = 1
        else:     
            d1[w] = (1 + math.log10(d1[w]))
    for w in d2.keys():
        if d2[w]==0:
            d2[w] = 1
        else:
            d2[w] = (1 + math.log10(d2[w]))

    return cosine_similarity(list(d1.values()), list(d2.values()))

def jaccard_correlation(s1,s2):
    count_intersection=0
    l = set(s1).union(set(s2))
    d1 = {}
    d2 = {}

    for e in l:
        d1[e] = 0
        d2[e] = 0
    for w in s1:
        d1[w]+=1;
    for w in s2:
        d2[w]+=1;
    for e in l:
        count_intersection+=min(d1[e],d2[e]);
    sim=count_intersection/(len(s1)+len(s2)-count_intersection)
    return sim


print("Static transator")
print("Enter choice")
print("1-Russian to Polish\n")
print("2-Polish to Russian\n")
print("0-exit()\n")

choice=int(input())
first=""
second=""

if(choice==1):
    weights_file = "model/ru_to_pl_20k.npy"
    first=input("\nFile to be transalted\n")
    second=input("\nConverted File to test\n")
elif(choice==2):
    weights_file = "model/pl_to_ru_20k.npy"
    first=input("\nFile to be transalted\n")
    second=input("\nConverted File to test\n")
else:
    print("Exit")			
    exit()

our_file_name = first
correct_file_name = second

## Open and Preprocess the files
our_file = open(our_file_name).read()
correct_file = open(correct_file_name).read()

our_file = preprocess(our_file)
correct_file = preprocess(correct_file)

t = np.load(weights_file, allow_pickle='TRUE').item()

print("\nNo. of word pairs:",len(t))

convert = {}
conv_prob = {}

for pair in t.keys():
    if not isinstance(pair, tuple):
        convert = t
        break
    if pair[1] not in conv_prob.keys():
        conv_prob[pair[1]] = 0
    if(conv_prob[pair[1]] < t[pair]):
        conv_prob[pair[1]] = t[pair]
        convert[pair[1]] = pair[0]

inverted_list = convert

##################### PROCESS SENTENCE PAIRS ##################

our_file = our_file.split('\n')
correct_file = correct_file.split('\n')

arr_cos = []
arr_jc = []

for i in range(len(our_file)):

    test = our_file[i]
    test_words = test.split(" ")
    conv = []
    for dw in test_words:
        if dw in inverted_list.keys():
            conv.append(inverted_list[dw])
    
    
    cor = correct_file[i]
    cor = cor.split(" ")
    
    
    cos_sim = cos(cor,conv)
    jc_sim = jaccard_correlation(cor,conv)
    arr_cos.append(cos_sim)
    arr_jc.append(jc_sim)

## Output 
print("Average Cosine Similarity",mean(arr_cos))
print("Average Jaccard Similarity",mean(arr_jc))

print("\nTranslation of:\n",our_file)
print("\nTranslation:\n",conv)
print("\nCorrect translation:\n",cor)
print()
########################################################