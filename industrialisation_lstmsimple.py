# -*- coding: utf-8 -*-
"""Industrialisation-lstmsimple

Original file is located at
    https://colab.research.google.com/drive/1zR3bAU6D25uEAJi6okvt7ZJxIHiaLbqM

# READ FILES
"""

from collections import defaultdict
import re
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords 

nltk.download('stopwords')



def load(fichier):
    '''
    Nettoie seulement les retours à la ligne puis split entre les parties du csv.
    Renvoie une liste d'offres
    Si le fichier est input, chaque offre a 2 parties: id, txt ([[id,txt],...])
    Sinon 5 parties: id, job, desc_entreprise, desc_poste, desc_profile ([[id,job,desc_entreprise,desc_poste,desc_profile],...])
    '''
    
    offers = []    # liste de toutes les offres dans le fichiers
    offer  = ''    # offre en cours de lecture


    with open(fichier, 'r') as file:
        for line in file:
            # on réalise tous les traitements nécessaires sur la ligne que l'on vient de lire
            offer_txt = line.split(",")
            
            # Le split généère plus de colonnes que nécessaire, car certains champs contiennent eux-mêmes des virgules
            # On le gère en distinguant les débuts d'offre
            
            if re.match("\"[0-9]+\"",offer_txt[0]):
                #si la ligne commence par un nombre c'est une nouvelle offre
                
                if offer != '':
                    #on ajoute l'offre  précedente à la liste des offres 
                    pred_offer = offer.replace("\n",'').replace("^M",'').replace("<", " <").replace(">", "> ").split('''","''')
                    # On splitte par "','" mais l'élément de début et celui de fin contiendra une extra "
                    # que l'on gère via la fonction gestion_extravirgules
                    offers.append(gestion_extravirgules(pred_offer))

                #la nouvelle offre
                offer = line
            
            #sinon on continue d'ajouter la ligne à l'offre
            else:
                offer += line

        #ajoute la derniere offre du fichier
        pred_offer = offer.replace("\n",'').replace("^M",'').replace("<", " <").replace(">", "> ").split('''","''')
        offers.append(gestion_extravirgules(pred_offer))
    return offers

def gestion_extravirgules(lstOffer):
    '''
    Fonction qui prend une liste en entrée et renvoie une nouvelle liste
    avec le premier (on le transforme en entier) et dernier élément modifié (on lui enlève un ")
    '''
    lst = []
    for i, column in enumerate(lstOffer):
        if i == 0:
            newcol = int(column.replace('"', ''))
        elif i == len(lstOffer) - 1:
            newcol = column.replace('"', '')
        else:
            newcol = column
        lst.append(newcol)
    return lst


def read(fichier_input):
    """
    Prends un fichier d'offres non decoupés et retourne une liste de tuples (id, offre)
    où offre est une sous-liste de phrases decoupés.
    """    
    
    inputs = []                          #liste des offres avec les phrases decoupés

    for offer in load(fichier_input):
        idx = offer[0]
        parts = offer[1]
        sentences = [sentence for sentence in re.split('(\. |</.{1,5}>|<br />)', parts)]  #offre splitté par phrase
        inputs.append((idx,sentences))
    
    return inputs


def clean(tagged):
    """
    Prends une liste d'offres [(phrases de l'offre,tag),...] où les phrases ne contenant que <br />, </p> ou . sont concaténées à la phrase précédente
    """
    tagged_clean = []
    
    for i, (sentence, label) in enumerate(tagged):
        
        if re.fullmatch("( |\. |</.{1,5}>|<br />)", sentence) or len(sentence)==0:
          tagged_clean[-1] = (tagged_clean[-1][0]+" "+sentence, tagged_clean[-1][1]) #Concaténation avec l'élément précédent
        else:
          tagged_clean.append((sentence, label))

    return tagged_clean

def train2tagged(fichier_train):
    """
    Prends un fichier d'offres deja decoupés et retourne une liste de tuples [(id, offre), ...]
    où offre est une sous-liste de tuples (phrase, tag)
    Les tags sont {0, 1, 2 ,3} pour job, desc_entreprise, desc_poste, desc_profile
    """

    train = load(fichier_train)

    train_tagged = []

    for offer in train:
        idx = offer[0]
        parts = offer[1:]
        tagged = [(sentence, i) for i in range(len(parts)) \
                                for sentence in re.split('(\. |</.{1,5}>|<br />)', parts[i] )]  #offre splitté par phrase
        train_tagged.append((idx,clean(tagged)))

    return train_tagged 


def tagged2output(tagged):
    """
    Transforme une liste d'offres taggés vers le format offers/output
    """
    output = []                               #liste des offres

    for idx, offer in tagged:
        output.append([idx])
        offer_dic = defaultdict(list)         #etape intermediaire qui va contenir toutes les parties de l'offre
        
        for sent, tag in offer:
            offer_dic[tag].append(sent)

        output[-1].extend(["".join(offer_dic[k]) for k in range(4)])

    return output



def accuracy_parts(preds_output, offers):
    """
    both tagged format
    """
    assert len(offers) == len(preds_output), "preds_output and golds_output not same size"

    corrects = defaultdict(float)
    totals   = defaultdict(int)

    for i,offer in enumerate(offers):
        #vérifie si les offres sont dans le même ordre
        assert offers[i][0] == preds_output[i][0], "offers not in same order"
        
        for j,(sentence, gold) in enumerate(offer[1]):
            if gold == preds_output[i][1][j][1]:
                corrects[gold]+=1
            totals[gold]+=1

    return {tag:100*corrects[tag]/totals[tag] for tag in corrects.keys()}


def accuracy_sentences(preds_tagged, golds_tagged):
    """
    precision des phrases taggés correspondant à la bonne partie de l'offre selon offers/output taggés 
    (= nb de phrases bien tagguées/nb de phrase du Gold (que l'on suppose ici = au nb de phrases de la prédiction))
    both tagged format
    """
    assert len(golds_tagged) == len(preds_tagged), "preds_tagged and golds_tagged not same size"

    total    = 0
    corrects = 0

    for i in range(len(golds_tagged)):
        #vérifie si les offres sont dans le même ordre
        assert golds_tagged[i][0]==preds_tagged[i][0], "offers not in same order"
        
        _        , tags  = zip(*preds_tagged[i][1])
        sentences, golds = zip(*golds_tagged[i][1])

        total    += len(sentences)
        corrects += sum([ypred==ygold for ypred, ygold in zip(tags,golds)])

    return 100*corrects/total


def order_eval(tagged):
    false_ordering = 0
    total          = 0.0

    for i in range(len(tagged)):
        _ , tags  = zip(*tagged[i][1])

        gov    = tags[0]
        total += len(tags)

        for k in range(1, len(tags)):
            if tags[k] < gov :
                false_ordering +=1
            else:
                 gov = tags[k] 

    return 100*false_ordering/total



def write(offers_tagged, name):
    ids, tagged_sentences = zip(*offers_tagged)

    #Generation of csv file for training
    df = pd.DataFrame({"idx": ids, "tagged_sentences": tagged_sentences})
    df.to_csv('name', sep = '\t', index = False)

path_test = "input.csv"
inputs  = load(path_test)
print("------INPUTS LOAD---------------")
print(inputs[0:2])
print(len(inputs[0]))
inputs=read(path_test)
print("------INPUTS READ---------------")
print(inputs[0:1])
print(len(inputs[0]))
path_output= "output.csv"
outputs = load(path_output)
print("------OUTPUTS LOAD---------------")
print(outputs[0:1])
print(len(outputs[0]))
print()
path_train = "offers.csv"
train   = load(path_train)
print("------TRAIN LOAD---------------")
print(train[0:1])
print(len(train[0]))

tagged = train2tagged(path_train)

print("------TRAIN TAGGED---------------")
print(tagged[len(tagged)-1])
print(len(tagged[0]))
res = tagged2output(tagged)

print("------TAGGED2OUTPUT---------------")
print(res[0:1])
print(len(res[0]))

print("-------ACCURACY PARTS---------------------")
print(accuracy_parts(tagged,tagged))


print("-------ACCURACY SENTENCES---------------------")
print(accuracy_sentences(tagged, tagged)) 

print("-------ORDER SENTENCES---------------------")
print(order_eval(tagged))

# Training


torch.manual_seed(0)
TARGETS = ['0','1','2','3']
TRAIN_SIZE = 45000
DEV_SIZE = 5000

tagged = train2tagged(path_train)


VECTORIZER = TfidfVectorizer(tokenizer=lambda x: x.split() , stop_words=stopwords.words("french"), min_df=500)
VECTORIZER.fit([part for offer in tagged2output(tagged) for part in offer[1:]])
print(len(VECTORIZER.vocabulary_))
print(VECTORIZER.vocabulary_)

print(VECTORIZER.idf_)

class OfferDataset(Dataset):
	"""
	This is a subclass of torch.utils.data Dataset and it implements 
	methods that make the dataset compatible with pytorch data utilities, notably the DataLoader
	"""
	def __init__(self,datalines):
		self.xydata = datalines
	
	def __len__(self):              #API requirement
		return len(self.xydata)
	
	def __getitem__(self,idx):      #API requirement
		return self.xydata[idx]



#Loading training set datas
def load_data_set(offers_tagged):
	"""
	Loads a dataset as a list of tuples: (text,label)
	Args:
	   filename (str): the dataset filename 
	Returns:
	   A pytorch compatible Dataset object
	   list of tuples.
	"""
	xydataset = [ ]
	for idx,text in offers_tagged:
		xydataset.append( (idx,text) )
	return OfferDataset(xydataset)


full_train_set = load_data_set(tagged)
print('Loaded %d examples for training. '%(len(full_train_set)))



#Splitting training datas : 80% as train_set and 20% as dev_set (validation set)
TRAIN_SET, DEV_SET = torch.utils.data.random_split(full_train_set, [TRAIN_SIZE, DEV_SIZE])
#TRAIN_SET, DEV_SET = full_train_set[:500],full_train_set[500:550]

#Maps words to integers
def make_w2idx(dataset):
	vocab = set([])
	for sentence,tag in dataset:
		words = sentence.split()
		vocab.update(words)
	return dict(zip(vocab,range(len(vocab))))   


def vectorize_text(sentence,vectorizer):
	# encode document
    vector = vectorizer.transform(sentence)

	#transform
    i = torch.LongTensor([vector.indices])
    v = torch.FloatTensor(vector.data)
    shape = vector.shape
    return torch.sparse.FloatTensor(i, v, torch.Size([shape[1]])).to_dense().view(shape)

def vectorize_target(Y):
	return  torch.LongTensor([int(Y)])



"""-------------------------------------------Tagging LSTM--------------------------------------------------"""
class LSTMOfferSplitter(nn.Module): 
	"""tagging de sequence d'une offre, chaque phrase est représentée par un bag of words"""

	def __init__(self, hidden_dim):
		super(LSTMOfferSplitter, self).__init__()
		self.hidden_dim = hidden_dim
		self.reset_structure(1,1,1)


	def reset_structure(self, hidden_dim, vocab_size, num_labels):

		# The LSTM takes bag of words representation as inputs, and outputs hidden states
		# with dimensionality hidden_dim.
		self.lstm = nn.LSTM(vocab_size, hidden_dim, bidirectional=True)
		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = nn.Linear(hidden_dim*2, num_labels)


	def forward(self, sentences):
		lstm_out, _      = self.lstm(sentences.view(len(sentences), 1, -1))
		tag_space        = self.hidden2tag(lstm_out.view(len(sentences), -1))
		tag_scores       = F.log_softmax(tag_space, dim=1)

		return tag_scores


	#Performs training on datas   
	def train(self, vectorizer, train_set, dev_set, learning_rate, epochs):
		start = time()
		
		self.reset_structure( self.hidden_dim, len(vectorizer.vocabulary_), len(TARGETS))
		self.dev_logloss = np.inf
		self.e           = 0
		
		loss_func   = nn.NLLLoss()
		#optimizer   = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
		optimizer  = optim.Adam(self.parameters(), lr=learning_rate)
		data_loader = DataLoader(train_set, batch_size=1, shuffle=True)
			
		for epoch in range(epochs):
			intermediate_time = time()
			global_logloss = 0.0
			for idx,tagged_text in data_loader: 

				X, Y = zip(*tagged_text)
				
				self.zero_grad()
				xvec = torch.cat([vectorize_text(sentence, vectorizer) for sentence in X],0)
				yvec = torch.cat([vectorize_target(label) for label in Y],0)
				output = self.forward(xvec)

				loss = loss_func(output, yvec)

				loss.backward()
				optimizer.step()
				global_logloss += loss.item()

			print("Epoch {}, mean cross entropy : train = {}".format(epoch,global_logloss/len(train_set)))
			dev_loss = self.dev(dev_set, vectorizer)
			print("    - mean cross entropy for dev = {}".format(dev_loss))

			if dev_loss < self.dev_logloss:
				self.e           = epoch
				self.dev_logloss = dev_loss
				torch.save(self, "lstmsimple2_param.pt")
			print("en",time()-intermediate_time)
			print()
   
		print(" ---- Best score on dev set = {} , epoch = {} ----".format(self.dev_logloss, self.e))
		print("Temps total", time()-start)

	#This function returns loss on validation set 
	def dev(self, dev_set, vectorizer):
		data_loader = DataLoader(dev_set, batch_size=1, shuffle=False)
		dev_logloss = 0.0
		loss_func = nn.NLLLoss()

		with torch.no_grad():
			for idx,tagged_text in data_loader: 

				X, Y = zip(*tagged_text)

				xvec = torch.cat([vectorize_text(sentence, vectorizer) for sentence in X],0)
				yvec = torch.cat([vectorize_target(label) for label in Y],0)
				output = self.forward(xvec)
				loss = loss_func(output, yvec)

				dev_logloss += loss.item()

		return dev_logloss/len(dev_set)


	def run(self, data_set, vectorizer):
		data_loader = DataLoader(data_set, batch_size=1, shuffle=False)
		preds = []

		with torch.no_grad():
			for idx,tagged_text in data_loader: 
				X, Y = zip(*tagged_text)

				xvec = torch.cat([vectorize_text(sentence,vectorizer) for sentence in X],0)
				yvec = torch.cat([vectorize_target(label) for label in Y],0)
				output = self(xvec)

				preds.append((idx[0], list(zip([sentence[0] for sentence in X],torch.argmax(output,dim=1).tolist())) ))
	
		return sorted(preds, key=lambda x: int(x[0]))



"""# Prédictions"""

lstm = torch.load("lstmsimple2_param.pt")

path_test = "output.csv"
test = train2tagged(path_test)
TEST_SET = load_data_set(test)
preds = lstm.run(TEST_SET, VECTORIZER)

path_testinput = "input.csv"
testinput = train2tagged(path_testinput)
IN_SET = load_data_set(testinput)
preds_input = lstm.run(IN_SET, VECTORIZER)

def clean_preds(preds_tagged):
	"""
	Prends une liste d'offres au format tagged
	et réordonne en place les tags si nécessaire.
	"""
	for idx,sentences_tagged in preds_tagged:
		gov = 1

		for k in range(1, len(sentences_tagged)):
			if sentences_tagged[k][1] < gov :
				sentences_tagged[k] = (sentences_tagged[k][0], gov)
			else:
				gov = sentences_tagged[k][1]
	return

def write_test_output(predictions):
	ids = []
	intitule = []
	des_ent = []
	des_po = []
	des_pro = []
	t0 = ""
	t1 = ""
	t2 = ""
	t3 = ""
	for k in range(len(predictions)):
		lst = list(predictions[k])
		ids.append(int(lst[0]))
		for prediction in lst[1]:
			if prediction[1] == 0:
				t0 = t0 + prediction[0]
			elif prediction[1] == 1:
				t1 = t1 + prediction[0]
			elif prediction[1] == 2:
				t2 = t2 + prediction[0]
			else:
				t3 = t3 + prediction[0]
		intitule.append(t0)
		des_ent.append(t1)
		des_po.append(t2)
		des_pro.append(t3)
		t0 = ""
		t1 = ""
		t2 = ""
		t3 = ""

	dfs = pd.DataFrame({"idx": ids, "intit": intitule, "des_ent":des_ent, "des_po":des_po, "des_pro":des_pro})
	dfs.to_csv('model_output.csv', sep = ',', index = False, header = False)


#le fichier gold est output donc test pour evaluation
#testinput n'est pas séparé, il n'y pas de tag, seulement pour la pred
print("pred_from_output accuracy_sentences {}%".format(accuracy_sentences(preds, test)))    #quand le fichier en input est output
print("pred_from_input accuracy_sentences {}%".format(accuracy_sentences(preds_input, test)))   #quand le fichier en input est input 
print()

print("pred_from_output accuracy_parts",accuracy_parts(preds, test))    #quand le fichier en input est output
print("pred_from_input accuracy_parts",accuracy_parts(preds_input, test))   #quand le fichier en input est input 
print()

print("pred_from_output false order {}%".format(order_eval(preds)))    #quand le fichier en input est output
print("pred_from_input false order {}%".format(order_eval(preds_input)))   #quand le fichier en input est input 
print()

print("----------Apres réordonnancement-------------------" )

clean_preds(preds)
clean_preds(preds_input)

print("pred_from_output accuracy_sentences {}%",accuracy_sentences(preds, test))    #quand le fichier en input est output
print("pred_from_input accuracy_sentences {}%",accuracy_sentences(preds_input, test))   #quand le fichier en input est input 
print()

print("pred_from_output accuracy_parts",accuracy_parts(preds, test)  )  #quand le fichier en input est output
print("pred_from_input accuracy_parts",accuracy_parts(preds_input, test))   #quand le fichier en input est input 
print()

print("pred_from_output false order {}%".format(order_eval(preds))  )  #quand le fichier en input est output
print("pred_from_input false order {}%".format(order_eval(preds_input)))   #quand le fichier en input est input 
print()

#creation d'un fichier model_output contenant les resultats sur le corpus de test
write_test_output(preds_input)

for k in range(len(test)):
	print(test[k])  #gold
	print(preds[k]) #pred
	print(preds_input[k]) #pred
	print()
