# -*- coding: utf-8 -*-
"""Industrialisation-mlpsimple

Original file is located at
    https://colab.research.google.com/drive/1_HQeBJN33WiPTfEW3mMLzYumrrIvVt9v

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
from collections import Counter
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
        #print(idx)
        #print(sentences)
        inputs.append((idx,sentences))
    
    return inputs

#read('input.csv')
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
        #print(len(tagged))
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
    accuracy du decoupage sur preds selon golds (= Somme(découpages corrects)/Somme(découpage du Gold) (que l'on suppose =nb de phrases de la prédiction))
    both output format
    """
    assert len(offers) == len(preds_output), "preds_output and golds_output not same size"

    total    = 0
    corrects = 0

    for i,offer in enumerate(offers):
        #vérifie si les offres sont dans le même ordre
        assert offers[i][0] == preds_output[i][0], "offers not in same order"
        
        for j, sentence in enumerate(offer[1:]):
            #Vérifie que le découpage est bien identique
            if preds_output[i][j]==offers[i][j]:
                corrects += 1
            total+=1

    return 100*corrects/total

def precision_parts(preds_output, offers):
    """
    precision du decoupage sur preds selon golds (= Somme(découpages corrects)/Somme(découpage de la prédiction)
    both output format. Ici on n'a pas besoin que |Gold|=|Predictions|
    """

    total    = 0
    corrects = 0

    for i,offer in enumerate(preds_output):
        #vérifie si les offres sont dans le même ordre
        assert offers[i][0] == preds_output[i][0], "offers not in same order"
        
        for j, sentence in enumerate(offer[1:]):
            #Vérifie que le découpage est bien identique
            if preds_output[i][j]==offers[i][j]:
                corrects += 1
            total+=1

    return 100*corrects/total

def rappel_parts(preds_output, offers):
    """
    precision du decoupage sur preds selon golds (= Somme(découpages corrects)/Somme(découpage du Gold))
    both output format. Ici on n'a pas besoin que |Gold|=|Predictions|
    """

    total=0
    corrects = 0

    for i,offer in enumerate(offers):
        #vérifie si les offres sont dans le même ordre
        assert offers[i][0] == preds_output[i][0], "offers not in same order"
        
        for j, sentence in enumerate(offer[1:]):
            #Vérifie que le découpage est bien identique
            if preds_output[i][j]==offers[i][j]:
                corrects += 1
            total+=1
    
    return 100*corrects/total

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

def rappel_sentences(preds_tagged, golds_tagged):
    """
    precision des phrases taggés correspondant à la bonne partie de l'offre selon offers/output taggés 
    (= nb de phrases bien tagguées/nb de phrase du Gold)
    Ici l'on n'a pas besoin que |Gold|=|Predictions|
    both tagged format
    """

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

def precision_sentences(preds_tagged, golds_tagged):
    """
    precision des phrases taggés correspondant à la bonne partie de l'offre selon offers/output taggés 
    (= nb de phrases bien tagguées/nb de phrase du Gold)
    Ici l'on n'a pas besoin que |Gold|=|Predictions|
    both tagged format
    """
    
    total    = 0
    corrects = 0

    for i in range(len(preds_tagged)):
        #vérifie si les offres sont dans le même ordre
        assert golds_tagged[i][0]==preds_tagged[i][0], "offers not in same order"
        
        preds        , tags  = zip(*preds_tagged[i][1])
        sentences, golds = zip(*golds_tagged[i][1])

        total    += len(preds)
        corrects += sum([ypred==ygold for ypred, ygold in zip(tags,golds)])

    return 100*corrects/total

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
print(accuracy_parts(res,train))
print("-------PRECISION PARTS---------------------")
print(precision_parts(res,train))
print("-------RAPPEL PARTS---------------------")
print(rappel_parts(res,train))
# L'accuracy n'est pas de 100% sur res/train car le découpage des parties d'offre n'est pas le même => Les parties dans offer commencent parfois par des .
# et la reconstruction n'est pas faite dans ce sens

print("-------ACCURACY SENTENCES---------------------")
print(accuracy_sentences(tagged, tagged)) 
print("-------PRECISION SENTENCES---------------------")
print(precision_sentences(tagged, tagged)) 
print("-------RAPPEL SENTENCES---------------------")
print(rappel_sentences(tagged, tagged))


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
#TRAIN_SET, DEV_SET = torch.utils.data.random_split(full_train_set, [TRAIN_SIZE, DEV_SIZE])
TRAIN_SET, DEV_SET = full_train_set[:500],full_train_set[1000:1500]

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



class MLPOfferSplitter(nn.Module):
	"""tagging de sequence d'une offre, chaue phrase est représenté par un bag of words"""

	def __init__(self, hidden_dim, tag_dim):
		super(MLPOfferSplitter, self).__init__()
		self.hidden_dim = hidden_dim
		self.tag_dim = tag_dim
		self.reset_structure(1, 1, 1, 1)

	def reset_structure(self, hidden_dim, tag_dim, vocab_size, num_labels):
		self.tagEmbedding = nn.Embedding(num_labels, tag_dim)

		# Input sont la concatenation de la reprensentation du label de la phrase precedente,la representation de cette phrase, et la phrase à tagger
		self.W = nn.Linear(2 * vocab_size + tag_dim, hidden_dim)

		# The linear layer that maps from hidden state space to tag space
		# prediction only on tag 1,2,3
		self.hidden2tag = nn.Linear(hidden_dim, num_labels)

	def forward(self, tag, sentences):

		input_mlp = torch.cat((self.tagEmbedding(tag), sentences), 1)
		hidden = torch.tanh(self.W(input_mlp))
		tag_space = self.hidden2tag(hidden)
		tag_scores = F.log_softmax(tag_space, dim=1)

		return tag_scores

	# Performs training on datas
	def train(self,  vectorizer, train_set, dev_set, learning_rate, epochs):

		self.reset_structure(self.hidden_dim, self.tag_dim, len(vectorizer.vocabulary_), len(TARGETS))
		self.dev_logloss = np.inf
		self.e = 0

		loss_func = nn.NLLLoss()
		optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		data_loader = DataLoader(train_set, batch_size=1, shuffle=True)

		for epoch in range(epochs):
			global_logloss = 0.0
			for idx, tagged_text in data_loader:
				pred_label = 0
				for k in range(1,len(tagged_text)):
					X, Y = tagged_text[k]
					self.zero_grad()
					phrase_prec = [tagged_text[k][0][0]]
					vec_prec = vectorize_text(phrase_prec, vectorizer)
					phrase_courante = [X[0]]
					vec_courant = vectorize_text(phrase_courante, vectorizer)
					xvec = torch.cat((vec_prec, vec_courant), 1)
					yvec = vectorize_target(Y)
					output = self.forward(vectorize_target(pred_label), xvec)

					pred_label = torch.argmax(output)

					loss = loss_func(output, yvec)

					loss.backward()
					optimizer.step()
					global_logloss += loss.item()

			print("Epoch {}, mean cross entropy : train = {}".format(epoch, global_logloss / len(
					[sentence for idx, tagged_text in train_set for sentence, label in tagged_text])))
			dev_loss = self.dev(dev_set,vectorizer)
			print("    - mean cross entropy for dev = {}".format(dev_loss))
			print()

			if dev_loss < self.dev_logloss:
					self.e = epoch
					self.dev_logloss = dev_loss
					torch.save(self, "mlpsimple_param.pt")
		print(" ---- Best score on dev set = {} , epoch = {} ----".format(self.dev_logloss, self.e))

	# This function returns loss on validation set
	def dev(self, dev_set,vectorizer):
		data_loader = DataLoader(dev_set, batch_size=1, shuffle=True)
		dev_logloss = 0.0
		loss_func = nn.NLLLoss()

		with torch.no_grad():
			for idx, tagged_text in data_loader:

				pred_label = 0
				for k in range(1,len(tagged_text)):
					X, Y = tagged_text[k]
					xvec = torch.cat((vectorize_text([tagged_text[k - 1][0][0]], vectorizer), vectorize_text([X[0]], vectorizer)), 1)
					yvec = vectorize_target(Y)
					output = self.forward(vectorize_target(pred_label), xvec)

					pred_label = torch.argmax(output)

					loss = loss_func(output, yvec)
					dev_logloss += loss.item()

		return dev_logloss / len([sentence for idx, tagged_text in dev_set for sentence, label in tagged_text])

	def run(self, data_set,vectorizer):
		data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

		preds = []

		with torch.no_grad():
			for idx, tagged_text in data_loader:
				seq = [0]
				pred_label = 0
				for k in range(1,len(tagged_text)):
					X, Y = tagged_text[k]

					xvec = torch.cat((vectorize_text([tagged_text[k - 1][0][0]], vectorizer), vectorize_text([X[0]], vectorizer)), 1)
					yvec = vectorize_target(Y)
					output = self.forward(vectorize_target(pred_label), xvec)

					pred_label = torch.argmax(output)
					seq.append(pred_label.item())

				preds.append((int(idx.squeeze()), list(zip([sentence[0] for sentence, label in tagged_text], seq))))

		return sorted(preds, key=lambda x: int(x[0]))
		
mlp = MLPOfferSplitter(100, 1000)
mlp.train(VECTORIZER, TRAIN_SET, DEV_SET, 0.0001, 5)

"""# Prédictions"""

mlp = torch.load("mlpsimple_param.pt")

path_test = "output.csv"
test = train2tagged(path_test)
TEST_SET = load_data_set(test)
preds = mlp.run(TEST_SET, VECTORIZER)

print("accuracy_sentences",accuracy_sentences(preds, test))
print()

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

#creation d'un fichier model_output contenant les resultats sur le corpus de test
write_test_output(preds_input)

for k in range(len(test)):
	print(test[k])  #gold
	print(preds[k]) #pred
	print()
