import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
#matplotlib inline
from torchtext import data, datasets
import pickle
from spectral_local_r2d2_path_settings import *
from tokenize_numeric import *
import datetime
from model_evaluation_support import *
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        a = self.decode(self.encode(src, src_mask), src_mask,
                    tgt, tgt_mask)
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        b = self.encoder(self.src_embed(src), src_mask)
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        c = x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        d = self.lut(x)
        #print(d.size, type(d))
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):   # d_ff = 2048
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        #print(i)
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


opts = [NoamOpt(512, 1, 4000, None),
        NoamOpt(512, 1, 8000, None),
        NoamOpt(256, 1, 4000, None)]
#plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
#plt.legend(["512:4000", "512:8000", "256:4000"])


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

crit = LabelSmoothing(5, 0, 0.1)
def loss(x):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                 ])
    #print(predict)
    return crit(Variable(predict.log()),
                 Variable(torch.LongTensor([1]))).data[0]


def data_gen(V, batch, nbatches, device):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        dd = np.random.randint(1, 100, size=(batch, 43))
        data = torch.from_numpy(dd)
        #data = torch.from_numpy(np.random.randint(1, 100, size=(batch, 43)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False).to(device)
        tgt = Variable(data, requires_grad=False).to(device)
        aaa = Batch(src, tgt, 0)
        yield Batch(src, tgt, 0)


def data_gen_1(V, xx, yy,  batch, nbatches, device):
    #np.putmask(xx, yy >= 100, 99)
    for i in range(nbatches):
        if (i+1)*nbatches <= len(xx):
            #ddd = x[i * batch: (i + 1) * batch, 0:V]
            dd = np.random.randint(1, 100, size=(batch, 43))
            #print(dd.size, type(dd))
            #print(ddd.size,type(ddd))
            #print(i, i*batch, (i+1)*batch)
            #print(np.max(dd))
            #print(np.max(ddd))

            data_x = torch.from_numpy(xx[i*batch: (i+1)*batch, 0:43])
            data_y = torch.from_numpy(yy[i*batch: (i+1)*batch, 0:43])
            #data_x = torch.from_numpy(dd)
            #data_y = torch.from_numpy(dd)

            #data_x[:, 0] = 1
            data_y[:, 0] = 1
            src = Variable(data_x, requires_grad=False).to(device)
            tgt = Variable(data_y, requires_grad=False).to(device)
            yield Batch(src, tgt, 0)


def data_gen_2(V, batch, nbatches, device):
    for i in range(nbatches):
        data_x = torch.from_numpy(np.random.randint(1, 100, size=(batch, 43)))
        data_y = torch.from_numpy(np.random.randint(1, 100, size=(batch, 43)))
        #data_x[:, 0] = 1
        data_y[:, 0] = 1
        src = Variable(data_x, requires_grad=False).to(device)
        tgt = Variable(data_y, requires_grad=False).to(device)
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data * norm


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

### Actual code starts here###
#pp_type = '_raw_'
pp_type = '_normalized_'
#pp_type = '_log_scale_'
#pp_type = '_norm_log_'
#pp_type = '_log_norm_'

model_path = return_model_path()
#model_name = 'transformer_103_22_2022_22_24_26'
#model_name = 'transformer_103_23_2022_20_54_50'
#model_name = 'transformer_103_27_2022_00_11_49'
model_name = 'transformer_1__normalized__101_04_08_2022_14_08_22'
#model = TheModelClass(*args, **kwargs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path + model_name).to(device)
model = model.to(device)


initial_data_train_fname, target_data_train_fname, initial_data_valid_fname, target_data_valid_fname, valid_data_index_fname = return_processed_file_names(pp_type)

pkl_file = open(initial_data_train_fname, 'rb')
initial_data_train= pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(target_data_train_fname, 'rb')
target_data_train= pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(initial_data_valid_fname, 'rb')
initial_data_test= pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(target_data_valid_fname, 'rb')
target_data_test= pickle.load(pkl_file)
pkl_file.close()

pkl_file = open(valid_data_index_fname, 'rb')
valid_data_index= pickle.load(pkl_file)
pkl_file.close()
#valid_data_index = valid_data_index.to_numpy()
X0 = initial_data_train.astype(np.float32)
Y0 = target_data_train.astype(np.float32)


bins_initial = return_range(100, initial_data_train, initial_data_test)
id_train = tokenize_numeric(initial_data_train, bins_initial)
id_test = tokenize_numeric(initial_data_test, bins_initial)
#dd = np.random.randint(1, 100, size=(20, 43))

bins_target = return_range(100, target_data_train, target_data_test)
tgt_train = tokenize_numeric(target_data_train, bins_target)
tgt_test = tokenize_numeric(target_data_test, bins_target)

V = 101

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
#model = make_model(V, V, N=2)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
model = model.to(device)

model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def test_model(test_x, test_y, y_hat,  test_index):
    print("Testing LSTM model!")
    #test_x = apply_log(test_x)
    #test_y = apply_log(test_y)
    #test_x = test_x.to_numpy()
    #test_y = test_y.to_numpy()

    rows, cols = test_y.shape
    #test_x = test_x.reshape(rows, cols, 1)

    for i in range(0, rows):
        #x_hat = test_x[i, :, :]
        #x_hat = x_hat[None, :]
        #y_hat = model.predict(test_x[i, :, :])
        print("Testing for datapoint", test_index[i])
        y_ampl = np.abs(np.max(test_y[i, :]) - np.min(test_y[i, :]))
        residuals_nn = (test_y[i, :] - y_hat[i, :]) / y_ampl

        fig2, axis = plt.subplots()
        #plt.plot(test_y[i, :], color='blue')
        plt.plot(y_hat[i, :], color='orange')
        #plt.title("validation for", str(test_index.loc[i]))
        plt.show()

        fig3, axis = plt.subplots()
        plt.plot(residuals_nn, color='green')
        plt.title("residuals for a small set")
        plt.show()
    print("Transformer had model has been tested! ")


def plot_wrapper(X, Y):
    rows, cols = X.shape
    fig = plt.figure()
    pred = []
    for i in range(0, 100):
        src = Variable(torch.LongTensor(X[i, 0:43].reshape([1, 43]))).to(device)
        #src = Variable(torch.Tensor(X[i,:,:].reshape(1,-1))).to(device)
        src_mask = Variable(torch.ones(1,1,cols)).to(device)


        #for j in range(0, cols):
        predd=greedy_decode(model, src, src_mask, max_len=cols, start_symbol=i).to('cpu')
        pred.append(predd.detach().numpy()) #.detach().numpy(), Y[i, j, 0], predd.detach().numpy() - Y[i, j, 0]
        #print(i)

        #pred = np.array(pred)
        #a = Y[i, :, 0]
        plt.plot(Y[i, :], color='blue', linewidth=0.1)
        print(i)
        b = predd.detach().numpy()
        #print(b)
        plt.plot(b[0,:], color='red', linewidth=0.1)
    plt.show()
    return pred


# for i in range(0, len(id_test)):
#     src = Variable(torch.LongTensor(id_test[i, 0:43].reshape([1, 43]))).to(device)
#     #src_1 = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) ).to(device)
#     src_mask = Variable(torch.ones(1, 1, 43) ).to(device)
#     print(greedy_decode(model, src, src_mask, max_len=43, start_symbol=1))
#     print("Reference:", tgt_test[i, 0:43])


pred = plot_wrapper(id_test, tgt_test)

# plot untokenized
testing_set_power = 100#len(pred)
#fig = plt.figure()
y_hat = []
goodness_descriptors = []
for i in range(0, testing_set_power):
    forecasted = token2numeric(pred[i], bins_target)
    mse, rho, max_test, max_hat, delta_max_val, delta_max_loc = goodness_descriptor(target_data_test[i, :], forecasted[0,:])
    goodness_descriptors.append([int(valid_data_index.loc[i]), mse, rho, max_test, max_hat, delta_max_val, delta_max_loc])
    #plt.plot(target_data_test[i, :], color='blue', linewidth=0.1)
    #plt.plot(forecasted[0, :], color='red', linewidth=0.1)

goodness_descriptors = np.array(goodness_descriptors)
columns = ['index', 'mse', 'rho', 'max_test', 'max_hat', 'delta_max_val', 'delta_max_loc']
goodness_descriptors = pd.DataFrame(goodness_descriptors, columns = columns)

time = datetime.datetime.now()
path = return_model_path()
data_name = 'validation_of_transformer_1_' + pp_type + '_' + str(V) + '_' + time.strftime("%m_%d_%Y_%H_%M_%S")+'.csv'
goodness_descriptors.to_csv(path + data_name, index = False)

print("That's all folks!!!")
#plt.show()


