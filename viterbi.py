import numpy as np
import collections as c

emision={}                  #emision probabilities
transition={}               #transition probabilities
bigrams=[]                  #tag bigrams

#adding line boundaries
lines=['null\t<s>']     
for line in open('wsj00-18.tag'):
    if "\t" in line:
        lines.append(line.strip())
    else:
        lines.extend(['null\t</s>','null\t<s>'])
del lines[-1]

#list of tags in the same sequence as that of wsj corpus
tags  = [l.split("\t")[1] for l in lines]     

#list of the set tags without word boundaries           
tl=[i for i in set(tags) if i not in ('</s>','<s>')]  
tl.insert(0, '<s>')

tc=c.Counter(tags)     #tag counts
lc= c.Counter(lines)   #word tag combination counts(line counts)
wordtags = [(l[0].split("\t")[0],l[0].split("\t")[1],l[1]) for l in lc.items()]  

#storing bigram of tags in a list
for i, t in enumerate(tags): 
    if i+1<len(tags):
        next=tags[i+1]
        bigrams.append(t+next)

# storing emision and transition probabilities
for t in set(tags):  
    emision[t]={w[0]:w[2]/tc[t] for w in wordtags if w[1]==t and w[0]!='null' }
    transition[t]={tn:bigrams.count(t+tn)/tc[t] for tn in set(tags)}        

tclen=len(tc)
tllen=len(tl)

def viterbi(sentence):
    
    slen=len(sentence)
    trelis=np.zeros((tclen,2+slen))     
    pointer=np.zeros((tclen,2+slen))   #back pointer trelis
    trelis[1,0]=1                      #probability of start state in the trelis
    im=np.zeros(2+tllen)               #intermediate array to store the calculated prababilities of transitioning from one state to another in each word
    
    #transitioning from <s> to the best probable state for the first word
    for j in range (1,tllen):           
        trelis[j,1]=transition['<s>'][tl[j]]*emision[tl[j]].get(sentence[0],0)
        pointer[j,1]=0          
    
    #transitions in between the start state and end state        
    for i in range (2,slen+1):  
        for j in range (1,tllen):   
            for k in range(1,tllen):
                im[k]=trelis[k,i-1]*transition[tl[k]][tl[j]]
            trelis[j,i]=max(im)*emision[tl[j]].get(sentence[i-1],0) 
            if trelis[j,i]>0:
                pointer[j,i]=np.argmax(im)
    
    #transitioning from the best probable state for the last word to </s>             
    for k in range(1,tllen):  
        im[k]=trelis[k,slen]*transition[tl[k]]['</s>']    
    trelis[46,slen+1]=max(im)
    pointer[46,slen+1]=np.argmax(im)
    
    #tracing back the backpointer trelis from </s> to <s> moving backwards
    trace=[tl[int(pointer[46,slen+1])]]  
    for t in range(slen,1,-1):         
        trace.append(tl[int(pointer[tl.index(trace[-1]),t])])
    trace.reverse()
    
    return trace
    
print (viterbi(['This','is','a','sentence','.']))
print (viterbi(['This','might','produce','a','result','if','the','system','works','well','.']))
print (viterbi(['Can','a','can','can','a','can','?']))
print (viterbi(['Can','a','can','move','a','can','?']))
print (viterbi(['Can','you','walk','the','walk','and','talk','the','talk','?']))


