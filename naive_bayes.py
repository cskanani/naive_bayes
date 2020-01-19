from collections import Counter
import re
import random

dataset_file = open('data/Dataset.txt', 'r')
data = []
for line in dataset_file.readlines():
    data.append(line[:-1].split('\t'))
random.shuffle(data) #shuffling data to select diffrent train and test case on each run

#function to pre_processcess a data sample
def pre_process(sample):
    url = '''httrue_positive_count\S + |@\S + |#\S + '''
    punctuations = '''[\,\.\"\!\:\;\-\–\—\?]'''
    suffixes = '''ly |est |er '''
    articles = ''' a | an |\s*the '''
    conjunctions = ''' after | although | as | as if | as long as | as much as | as soon as | as though | because | before | by the time | even if | even though | if | in order that | in case | lest | once | only if | provided that | since | so that | than | that | though | till | unless | until | when | whenever | where | wherever | while | for | and | nor | but | so | or | yet '''
    pronouns = ''' it | myself | those | them | anything | few | everybody | this | one | these | her | whomever | itself | hers | they | whatever | she | themselves | none | any | both | who | more | nobody | enough | ours |\s*i | whichever | you | all | ourselves | he | whose | another | noone | yourself | himself | anybody | what | each | some | something | herself | whoever | us | his | neither | such | other | someone | most | whom | others | mine | everyone | anyone | everything | little | either | we | theirs | me | nothing | that | many | him | several | somebody | yours | much | which '''
    sample = re.sub(url, " ", sample)
    sample = re.sub(conjunctions, " ", sample)
    sample = re.sub(articles, " ", sample)
    sample = re.sub(suffixes, " ", sample)
    sample = re.sub(punctuations, " ", sample)
    sample = re.sub("(\w + \'t)", " not ", sample)
    sample = re.sub("(\')", " ", sample)
    sample = re.sub(pronouns, " ", sample)
    positive_emojis = ''':‑\)\|:-]\|:-3\|:->\|8-\)\|:-\}\|:o\)\|:c\)\|:\^\)\|=]\|=\)\|:\)\|:]\|:3\|:>\|8\)\|:\}\|:‑D\|:D\|8‑D\|8D\|x‑D\|xD\|X‑D\|XD\|=D\|=3\|B\^D\|:-\)\)\|:'‑\)\|:'\)\|:‑O\|:O\|:‑o\|:o\|:-0\|8‑0\|>:O\|:-\*\|:\*\|:×\|;‑\)\|;\)\|\*-\)\|\*\)\|;‑]\|;]\|;\^\)\|:‑,\|;D\|:‑P\|:P\|X‑P\|XP\|x‑p\|xp\|:‑p\|:p\|:‑Þ\|:‑Þ\|:‑þ\|:þ\|:Þ\|:Þ\|:‑b\|:b\|d:\|=p\|>:P\|O:‑\)\|O:\)\|0:‑3\|0:3\|0:‑\)\|0:\)\|0;\^\)\|\|;‑\)\|:‑J\|#‑\)\|%‑\)\|%\)\|<3\|@\};-\|@\}->--\|@\}‑;‑'‑‑‑\|@>‑‑>‑‑'''
    negative_emojis = ''':‑\(\|:\(\|:‑c\|:c\|:‑<\|:<\|:‑\[\|:\[\|:-\|\|\|>:\[\|:\{\|:@\|>:\(\|:'‑\(\|:'\(\|D‑':\|D:<\|D:\|D8\|D;\|D=\|DX\|:‑/\|:/\|:‑\.\|>:\\\|:L\|=L\|:S\|:‑\|\|:\|\|:‑X\|:X\|:‑#\|:#\|:‑&\|:&\|>:‑\)\|>:\)\|\}:‑\)\|\}:\)\|3:‑\)\|3:\)\|>;\)\|',:-l\|',:-\|\|>_>\|<_<\|<\|</3'''
    sample = re.sub(positive_emojis, " smile_positive ", sample)
    sample = re.sub(negative_emojis, " smile_negative ", sample)
    sample = re.sub('\s+', " ", sample)
    return sample

for x in data:
    x[1] =  pre_process(x[1].lower())
    
word_count = {}
positive_word_count = 0
negative_word_count = 0
positive_sample_count = 0
negative_sample_count = 0

# dividing data into train and test split
data_length = len(data)
train = data[:int(.75*data_length)]
test = data[int(.75*data_length):]

#training model using train data
for x in train:
    if(x[0] == '0'):
        negative_sample_count += 1
    else:
        positive_sample_count += 1
    
    word_list = list(filter(bool, x[1].split(' ')))
    for word in word_list:
        if(word in word_count):
            if(x[0] == '0'):
                negative_word_count += 1
                word_count[word][0] +=1
            else:
                positive_word_count += 1
                word_count[word][1] +=1
        else:
            if(x[0] == '0'):
                negative_word_count += 1
                word_count[word] = [1, 0]
            else:
                positive_word_count += 1
                word_count[word] = [0, 1]
                

#finding probability and applying laplace smoothing
probabilities = {}
al = 1
div = negative_sample_count + positive_sample_count
for x in word_count:
    probabilities[x] = [(word_count[x][0] + al)/(negative_word_count * div),
                        (word_count[x][1] + al) / (positive_word_count * div)]

#predicting class of test data
predictions = []
negative_probability = negative_sample_count / (negative_sample_count + positive_sample_count)
positive_probability = positive_sample_count / (negative_sample_count + positive_sample_count)
for x in test:
    word_list = list(filter(bool, x[1].split(' ')))
    for word in word_list:
        if(word in probabilities):
            negative_probability *= probabilities[word][0]
            positive_probability *= probabilities[word][1]
    
    if(positive_probability >= negative_probability):
        predictions.append('1')
    else:
        predictions.append('0')

#calculating accuracy for test data
true_positive_count = 0
false_positive_count = 0
true_negative_count = 0
false_negative_count = 0
for x, y in zip(test, predictions):
    if(x[0] != y):
        if(y == '1'):
            false_positive_count += 1
        else:
            false_negative_count += 1
    else:
        if(y == '1'):
            true_positive_count += 1
        else:
            true_negative_count += 1


print('\nModel Evaluation')
total_samples = true_positive_count + false_positive_count + true_negative_count + false_negative_count
accuracy = (total_samples - (false_positive_count + false_negative_count)) / total_samples
precision = true_positive_count / (true_positive_count + false_positive_count)
recall = true_positive_count / (true_positive_count + false_negative_count)
f1_score = 2 * precision * recall / (precision + recall)

print('Accuracy :', accuracy)
print('Precision :', precision)
print('Recall :', recall)
print('F1 Score :', f1_score)
