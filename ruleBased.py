import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans
import pandas as pd

nlp = spacy.load('en_core_web_sm')

causalVerbs = ['actuate','decide', 'increase',
                'affect','decrease','influence',
                'breed', 'determine','create',
                'cause','effect','reduce',
                'compel','impose','result']

relC = ['after','as','because','since']

pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'AUX', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]

def phrasesExtraction(sentence):
    '''
    This function is responsible to extract
    the Noun and Verb Phrases from the given 
    sentence
    '''
    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", None, pattern)

    doc = nlp(sentence) 
    # call the matcher to find matches 
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    # print('NP: ',list(doc.noun_chunks))
    # print ('VP: ',filter_spans(spans))   

    for np in list(doc.noun_chunks):
        sentence = sentence.replace(str(np),'NP')
    for vp in filter_spans(spans):
        sentence = sentence.replace(str(vp), 'VP')
    sentence = sentence.replace('(NP)','NP')
    sentence = sentence.replace('(VP)','VP')
    # print (sentence)
    return sentence

def ruleBasedExtractor(text):
    text = phrasesExtraction(text)

    pattern1 = re.compile('VP (after|before|as|because|since).*(after|before|as|because|since) VP')
    pattern2 = re.compile('NP VP NP')
    pattern3= re.compile('VP (after|before|as|because|since) NP .* (after|before|as|because|since) NP VP')

    if pattern1.search(text)== None:
        return 0 # stands for No causal relationship
    else:
        sentences = [str(s) for s in nlp(text).sents]
        if len(sentences) == 1:
            return 1 # stands for E1 precedes E2 
        else:
            tokens = [word.text for word in sentences[1:]]
            
            for verb in causalVerbs:
                if verb in tokens:
                    return 2 # stands for E2 precedes E1
            return 1 #stands for E1 precedes E2

def ruleBasedModel(InFile, OutFile):

    data = pd.read_csv(InFile, sep='\t', header=None, names=['text', 'expected_label'])
    data.expected_label.replace({'None': 0,
                        'E1 precedes E2': 1,
                        'E2 precedes E1': 2,
                        }, inplace=True)
    data.text = [re.sub(r'\s+', ' ', text).strip() for text in data.text.values]
    data.text = [x.lower() for x in data.text.values]

    data['predicted_label'] = data['text'].apply(ruleBasedExtractor)

    data.to_csv(OutFile,sep='\t',header=False)

