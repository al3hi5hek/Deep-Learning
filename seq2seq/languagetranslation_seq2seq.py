from io import open
import unicodedata
import string
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

 
SOS_token = 0
EOS_token = 1

# clas to hold langauge specific information
# Maps words to theri numeric id's and vice versa..for the input langauge and the output language

class Lang:
    def __init__(self , name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: 'SOS' , 1 : 'EOS'} # map numeric indices to their corresponding words 
        self.n_words = 2
        
    def addSentence(self , sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self , word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words # map word to index
            self.word2count[word] = 1 # track count of words
            self.index2word[self.n_words] = word # map index to word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
 

def normalizeString(s):
    
    '''
    convert unicode characters to ASCII format
    '''
    s = s.lower().strip()
    s = "".join(
    char for char in unicodedata.normalize('NFD',s) if unicodedata.category(char) != 'Mn')
    
    s = re.sub(r"([.!?])",r" \1" , s) # replace ?.! with a standard character
    s = re.sub(r"[^a-zA-Z.!?]+",r" ",s) # substitute all alpha-numeric characters with blanks
    return s

# Initialize lamg classes , one for source and one for target language

# preparing sentence pairs
def readLangs(lang1 , lang2 , reverse = False):
    
    print("Reading Lines...")
    
    lines = open(r'C:\Users\abhishek.shetty\Desktop\Abhishek\Courses\Plural\NLP with Pytorch\natural-language-processing-pytorch\datasets\data\%s-%s.txt' %(lang1 , lang2) , encoding = 'utf-8').read().strip().split('\n')
    
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang , output_lang , pairs    

MAX_LENGTH = 10 # can take a long time , limit the length of the sentences you work with to 10
eng_prefixes = ('i am' , "i m ","he is ", "he s ", "she is", "she s ", "you are", "you re ", "we are", "we re ", "they are", "they re ") # limit english sentences that start with these prefixes to reduce training data

def filterPairs(pairs):
    '''
    keep only those pairs where len of german input sentences and english output sentences is less than max_length 
    and englishs entences start with above prefixes
    '''
    return [p for p in pairs 
            if 
            len(p[0].split(' ')) < MAX_LENGTH and 
            len(p[1].split(' ')) < MAX_LENGTH and 
            p[1].startswith(eng_prefixes)]


def prepareData(lang1 , lang2 , reverse = False):
    input_lang , output_lang , pairs = readLangs(lang1 , lang2 , reverse) # read data
    print('Read %s sentence pairs' % len(pairs))
    
    pairs = filterPairs(pairs) # filtr pairs based on previous conditions
    print("trimmed to %s sentence pairs" % (len(pairs)))
    
    for pair in pairs: # add words to our dicitonaries and map them to indices and vice versa
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    print("counted words:")
    print(input_lang.name , input_lang.n_words)
    print(output_lang.name , output_lang.n_words)
    
    return input_lang , output_lang , pairs


input_lang , output_lang , pairs = prepareData('eng','deu' , reverse = True)
import random
print(random.choice(pairs))
    

# Encoder and Decoder architecture

# 2 neural networks , encoder and decoder

class EncoderRNN(nn.Module):
    
    def __init__(self , input_size , hidden_size):
        
        super(EncoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size , hidden_size) # one hot vectors converted to dense embeddings
        self.gru = nn.GRU(hidden_size , hidden_size)
        
    def forward(self , input , hidden):
        
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        output , hidden = self.gru(output , hidden)
        
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size)
        
      

class DecoderRNN(nn.Module):
    
    def __init__(self , hidden_size , output_size):
        
        super(DecoderRNN , self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size , hidden_size)
        self.gru = nn.GRU(hidden_size , hidden_size)
        self.out = nn.Linear(hidden_size , output_size)
        self.softmax = nn.LogSoftmax(dim = 1)
        
    def forward(self , input , hidden):
        
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        output , hidden = self.gru(output , hidden)
        output = self.softmax(self.out(output[0]))
        return output , hidden
    
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size)
    
    
# convert a sentence of words to a corresponding tensor representation of one hot encodings
def tensorFromSentence(lang , sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')] # will convert a sentence to indexes by referring to the langauge class mappings that we created earlier
    indexes.append(EOS_token)# we added a end of sentence token to the end of this list
    return torch.tensor(indexes , dtype = torch.long).view(-1,1) # shape of sentence_length , 1

# we need both sentence from source and target language to be represented in tensor formnd a target tensor

def tensorsFromPair(pair):
    
    input_tensor = tensorFromSentence(input_lang , pair[0])
    target_tensor = tensorFromSentence(output_lang , pair[1])
    
    return (input_tensor , target_tensor)


def train(input_tensor , target_tensor , encoder  ,decoder , encoder_optimizer , decoder_optimizer , criterion):
    
    encoder_hidden = encoder.initHidden() # hidden state all zeros
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0) # source sentence pair
    target_length = target_tensor.size(0)
    
    loss = 0
    
    for ei in range(input_length): # iterate over all words in input sentence
        encoder_output , encoder_hidden = encoder(input_tensor[ei],encoder_hidden) 
        # at each time instant, the input is the current word and hidden state from previous word
    
    
    decoder_input = torch.input([[SOS_token]]) # first input to decoder is start of sentence token
    decoder_hidden = encoder_hidden # final hidden state of encoder - all information from source langauge sentence is encapsulated in this final hidden state from encoder
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        
        for di in range(target_length): # iterate over all words in target sentence
            decoder_output , deocder_hidden = decoder(deocder_input , decoder_hidden)
            loss += criterion(decoder_output , target_tensor[di])
            decoder_input = target_tensor[di] # due to teacher forcing , the next input is the actual translated word and not the predicted output word
            
    else:
        
        for di in range(target_length): # iterate over all words in target sentence
            decoder_output , deocder_hidden = decoder(deocder_input , decoder_hidden)
            topv , topi = decoder_output.topk(1) # to find what word was predicted by the decoder at this time instance
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output , target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
                
    loss.backward()
    encoder_optimizer.step()
    decoder.optimizer.step()
    
    return loss.item() / target_length

plot_losses = []
print_loss_total = 0
plot_loss_total = 0
hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words , hidden_size) # input size of encoder is number of words in source langauge vocabulary
decoder1 = DecoderRNN(hidden_size , output_lang.n_words)  # outut size of decoder is number of words in target langauge vocabulary

# optimizers
encoder_optimizer = optim.SGD(encoder1.parameters() , lr = 0.01)
decoder_optimizer = optim.SGD(decoder1.parameters() , lr = 0.01)

training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(30000)]
criterion = nn.NLLLoss()


for iter in range(1 , 30001):
    
    training_pair = training_pairs[iter-1]
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]
            
    loss = train(input_tensor , target_tensor , encoder1 , decoder1 , encoder_optimizer , decoder_optimizer , criterion)
    print_loss_total += loss
    plot_loss_total += loss
    
    if iter % 1000 == 0:
        print_loss_avg = print_loss_total / 100
        print_loss_total = 0
        print('iteration - %d loss - %4f' %(iter , print_loss_avg))
        
    if iter % 100 == 0:
        plot_loss_avg = plot_loss_total / 100
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        
fig , ax = plt.subplots(figsize = (15,8))
loc = ticker.MultipleLocator(base=0.2)
ax,yaxis.set_major_locator(loc)
plt.plot(plot_losses)



# Translating Sentences

def evaluate(encoder , deocder , sentence):
    
    with torch.no_grad():
        
        input_tensor -= tensorFromSentence(input_lang , sentence)
        input_length = input_tensor.size(0)
        
        encoder_hidden = encoder.initHidden()
        
        for ei in range(input_length):
            encoder_output , encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            
        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        
        for di in range(MAX_LENGTH):
            decoder_output , decoder_hidden = decoder(decoder_input , decoder_hidden)
            topv , topi = decoder.output.data.topk(1) # output probabilities , output wi
            
            if topi.item() == 'EOS_token':
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang ,,index2word[topi.item()])
                
            decoder_input = topi.squeeze().detach()
            
        return decoded_words 
    
    
for in range(10)
    pair = random.choice(pairs)
    print('>' , pair[0])
    print('=' , pair[1])
    
    output_words = evaluate(encoder1 , decoder1 , pair[0])
    output_sentence = " ".join(output_words)
    
    print('<' , output_snetence)
    print('')
                
                
                
                
input_sentence = 'es tut mir sehr leid'
output_words = evaluate(encoder1  ,decoder1 , input_sentence)
print('input =',input_sentence)
pirnt('output =', ' '.join(output_words))