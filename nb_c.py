import re
from collections import Counter
import random

f = open('data/Dataset.txt','r')
dt = []
for x in f.readlines():
    dt.append(x[:-1].split('\t'))
#print(dt)
random.shuffle(dt) #shuffling data to select diffrent train and test case on each run

#function to preprocess data
def prepro(temps):
    urle = '''http\S+|@\S+|#\S+'''
    punct = '''[\,\.\"\!\:\;\-\–\—\?]'''
    suff = '''ly |est |er '''
    artcl = ''' a | an |\s*the '''
    con = ''' after | although | as | as if | as long as | as much as | as soon as | as though | because | before | by the time | even if | even though | if | in order that | in case | lest | once | only if | provided that | since | so that | than | that | though | till | unless | until | when | whenever | where | wherever | while | for | and | nor | but | so | or | yet '''
    pron = ''' it | myself | those | them | anything | few | everybody | this | one | these | her | whomever | itself | hers | they | whatever | she | themselves | none | any | both | who | more | nobody | enough | ours |\s*i | whichever | you | all | ourselves | he | whose | another | noone | yourself | himself | anybody | what | each | some | something | herself | whoever | us | his | neither | such | other | someone | most | whom | others | mine | everyone | anyone | everything | little | either | we | theirs | me | nothing | that | many | him | several | somebody | yours | much | which '''
    #temps = re.sub(urle+'|'+con+'|'+artcl+'|'+suff+'|'+punct, " ", temps)
    temps = re.sub(urle, " ", temps)
    temps = re.sub(con, " ", temps)
    temps = re.sub(artcl, " ", temps)
    temps = re.sub(suff, " ", temps)
    temps = re.sub(punct, " ", temps)
    temps = re.sub("(\w+\'t)"," not ",temps)
    temps = re.sub("(\')"," ",temps)
    temps = re.sub(pron, " ", temps)
    spre = ''':‑\)\|:-]\|:-3\|:->\|8-\)\|:-\}\|:o\)\|:c\)\|:\^\)\|=]\|=\)\|:\)\|:]\|:3\|:>\|8\)\|:\}\|:‑D\|:D\|8‑D\|8D\|x‑D\|xD\|X‑D\|XD\|=D\|=3\|B\^D\|:-\)\)\|:'‑\)\|:'\)\|:‑O\|:O\|:‑o\|:o\|:-0\|8‑0\|>:O\|:-\*\|:\*\|:×\|;‑\)\|;\)\|\*-\)\|\*\)\|;‑]\|;]\|;\^\)\|:‑,\|;D\|:‑P\|:P\|X‑P\|XP\|x‑p\|xp\|:‑p\|:p\|:‑Þ\|:‑Þ\|:‑þ\|:þ\|:Þ\|:Þ\|:‑b\|:b\|d:\|=p\|>:P\|O:‑\)\|O:\)\|0:‑3\|0:3\|0:‑\)\|0:\)\|0;\^\)\|\|;‑\)\|:‑J\|#‑\)\|%‑\)\|%\)\|<3\|@\};-\|@\}->--\|@\}‑;‑'‑‑‑\|@>‑‑>‑‑'''
    snre = ''':‑\(\|:\(\|:‑c\|:c\|:‑<\|:<\|:‑\[\|:\[\|:-\|\|\|>:\[\|:\{\|:@\|>:\(\|:'‑\(\|:'\(\|D‑':\|D:<\|D:\|D8\|D;\|D=\|DX\|:‑/\|:/\|:‑\.\|>:\\\|:L\|=L\|:S\|:‑\|\|:\|\|:‑X\|:X\|:‑#\|:#\|:‑&\|:&\|>:‑\)\|>:\)\|\}:‑\)\|\}:\)\|3:‑\)\|3:\)\|>;\)\|',:-l\|',:-\|\|>_>\|<_<\|<\|</3'''
    temps = re.sub(spre," smile_positive ",temps)
    temps = re.sub(snre," smile_negative ",temps)
    temps = re.sub('\s+'," ",temps)
    return temps

for x in dt:
    x[1] =  prepro(x[1].lower())
    
wordcnt = {}
pwc = 0
nwc = 0
psc = 0
nsc = 0
n = len(dt)
train = dt[:int(.75*n)]
test = dt[int(.75*n):]

#training model using train data
for x in train:
    if(x[0] == '0'):
        nsc += 1
    else:
        psc += 1
    
    wl = list(filter(bool,x[1].split(' ')))
    for y in wl:
        if(y in wordcnt):
            if(x[0] == '0'):
                nwc += 1
                wordcnt[y][0]+=1
            else:
                pwc += 1
                wordcnt[y][1]+=1
        else:
            if(x[0] == '0'):
                nwc += 1
                wordcnt[y] = [1,0]
            else:
                pwc += 1
                wordcnt[y] = [0,1]
                

#finding probability and applying laplace smoothing
prbdict = {}
al = 1
div = nsc+psc
for x in wordcnt:
    prbdict[x] = [(wordcnt[x][0]+al)/(nwc*div),(wordcnt[x][1]+al)/(pwc*div)]
#print(prbdict)

#predicting class of test data
ou = []
ncp = nsc/(nsc+psc)
pcp = psc/(nsc+psc)
for x in test:
    #print(x)
    nv= ncp
    pv = pcp
    wl = list(filter(bool,x[1].split(' ')))
    
    for y in wl:
        if(y in prbdict):
            nv *= prbdict[y][0]
            pv *= prbdict[y][1]
    
    if(pv >= nv):
        ou.append('1')
    elif(pv < nv):
        ou.append('0')

#calculating accuracy for test data
tp = 0
fp = 0
tn = 0
fn = 0
print('\nProcessed Misclassified Samples')
for x,y in zip(test,ou):
    if(x[0] != y):
        print(x,y)
        if(y == '1'):
            fp += 1
        else:
            fn += 1
    else:
        if(y == '1'):
            tp += 1
        else:
            tn += 1


print('\nModel Evaluation')
lent = tp+fp+tn+fn
acc = (lent-(fp+fn))/lent
pre = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = 2*pre*rec/(pre+rec)

print('Accuracy :',acc)
print('Precision :',pre)
print('Recall :',rec)
print('F1 Score :',f1)
