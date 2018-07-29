# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:31:43 2018

@author: bager
"""
#Imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from pandas_datareader import data
import tensorflow as tf
#from tensorflow.contrib.rnn import LSTMCell
tf.logging.set_verbosity(tf.logging.ERROR)
from time import time
from matplotlib.colors import hsv_to_rgb
from scipy.stats import ks_2samp
import matplotlib.mlab as mlab
######################################################################################################################
######################################################################################################################

# STEP 1 SIMULATE LOG RETURNS VIA MONTE CARLO SIMULATIONS (GENERATE SOME THE DATA USED TO LEARN THE GAN)


"""
we will do this by usual Monte Carlo simulation to look at the potential evolution of asset prices
over time, assuming they are subject to daily returns that follow a normal distribution. 
To set up our simulation, we need to estimate the expected level of return (mu) and volatility (vol) of the stock
"""

# ESTIMATION OF THE MEAN AND VOLATILITY

T = 9 #Number of trading days

#download Apple price data into DataFrame
apple = data.DataReader('AFI.AX', 'yahoo',start='1/1/2013')

#calculate the compound annual growth rate (CAGR) which 
#will give us our mean return input (mu) 
days = (apple.index[-1] - apple.index[0]).days
cagr = ((((apple['Adj Close'][-1]) / apple['Adj Close'][1])) ** (365.0/days)) - 1
print ('CAGR =',str(round(cagr,4)*100)+"%")
mu = cagr
 
#create a series of percentage returns and calculate the annual volatility of returns
apple['Returns'] = apple['Adj Close'].pct_change()
vol = apple['Returns'].std()*math.sqrt(T)
print ("Annual Volatility =",str(round(vol,4)*100)+"%")


# GENERATION OF THE DATA USED TO TRAIN THE GAN

#Define Variables
S = apple['Adj Close'][-1] #starting stock price (i.e. last available real stock price)

n_samples = 50000
seq_length = 10 # change line 202 and 227 as well

#log returns
def sample_data(n_samples=n_samples):
    vectors = []
    ###### MONTE CARLO #######
    for i in range(n_samples):
    
        #create list of daily returns using random normal distribution
        daily_returns=np.random.normal(mu/T,vol/math.sqrt(T),seq_length) #T+1 since we work with the vectors = [] and do not initilase it by [S]
        vectors.append(np.log(daily_returns+1))
    
    dataset = np.array(vectors)
    #dataset = np.expand_dims(dataset, axis=0)
    #dataset = dataset.reshape(38, 253, 1)
    
    dataset.reshape(-1, seq_length, 1)
    
    return dataset

###########################################################################################################################
###########################################################################################################################


# STEP 2 : DEFINE THE NETWORKS FOR THE GAN

# --- to do with training --- #

# function for getting one mini batch
def get_batch(samples, batch_size, batch_idx):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos]


tf.reset_default_graph()

print("loading data")

samples = sample_data(n_samples = n_samples)


print("data loaded")

#training configuration

print("loading settings")

    
learning_rate = 0.001
lr = learning_rate
batch_size = 28
#cond = 0
latent_dim = 20
num_epochs = 50
vis_freq = 2
labels = None 
D_rounds = 3
G_rounds = 1
hidden_units_g = 150
hidden_units_d = 150
num_generated_features = 1


CG = tf.placeholder(tf.float32, [batch_size, seq_length]) #Placeholder 0 (shape: (50,253)) is it 0 or seq_length
CD = tf.placeholder(tf.float32, [batch_size, seq_length]) #Placerholder 1
Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim]) #Placeholder 2
W_out_G = tf.Variable(tf.truncated_normal([hidden_units_g, num_generated_features]))
b_out_G = tf.Variable(tf.truncated_normal([num_generated_features]))

X = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
W_out_D = tf.Variable(tf.truncated_normal([hidden_units_d,1]))
b_out_D = tf.Variable(tf.truncated_normal([1]))


print("settings loaded")


# --- to do with latent space --- #

def sample_Z(batch_size, seq_length, latent_dim):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    return sample



#Define the generator and the discsriminator networks using LSTM cells

def generator(z, c):
    with tf.variable_scope("generator") as scope:
        
        # each step of the generator takes a random seed + the conditional embedding
        repeated_encoding = tf.tile(c,[1, tf.shape(z)[1]])
        repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1],10]) #SHOULD BE 0 INSTEAD OF 253 BUT THE ERROR
        generator_input = tf.concat([repeated_encoding, z], 2)
        
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_g, state_is_tuple=True)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                sequence_length=[seq_length]*batch_size,
                inputs=generator_input)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d


def discriminator(x, c, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        # correct?
        if reuse:
            scope.reuse_variables()
             
        # each step of the generator takes one time step of the signal to evaluate + 
        # its conditional embedding  
        repeated_encoding = tf.tile(c,[1, tf.shape(x)[1]])
        repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(x)[0], tf.shape(x)[1],10]) #SHOULD BE 0 INSTEAD OF 253 BUT THE ERROR
        decoder_input = tf.concat([repeated_encoding, x], 2)
        
        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, state_is_tuple=True)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell=cell,
                dtype=tf.float32,
                inputs=decoder_input)
        rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D
        output = tf.nn.sigmoid(logits)
    return output, logits


# Define the loss funtion for the GAN

G_sample = generator(Z, CG)
D_real, D_logit_real = discriminator(X, CD)
D_fake, D_logit_fake = discriminator(G_sample, CG, reuse=True)

generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                     labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                     labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                labels=tf.ones_like(D_logit_fake)))


# Define the optimisers (GD for the discriminator and Adam for generator)
    
D_solver = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(D_loss, var_list=discriminator_vars)
#D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=generator_vars)
#G_solver = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(G_loss, var_list=discriminator_vars)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=generator_vars)


##################################################################################################################

# Train the GAN on the Monte Carlo simulations

# Starting a tensorflow session

sess = tf.Session()
sess.run(tf.global_variables_initializer())
vis_Z = sample_Z(batch_size, seq_length, latent_dim)


t0 = time()

def train_generator(batch_idx, offset):
    # update the generator
    for g in range(G_rounds):
        #X_mb = get_batch(samples, batch_size, batch_idx + g + offset)
        Y_mb = get_batch(samples, batch_size, batch_idx + g + offset)
        _, G_loss_curr = sess.run([G_solver, G_loss],
                                  feed_dict={CG:Y_mb,
                                             Z: sample_Z(batch_size, seq_length, latent_dim)})
    return G_loss_curr


def train_discriminator(batch_idx, offset):
    # update the discriminator
    for d in range(D_rounds):
    # using same input sequence for both the synthetic data and the real one,
    # probably it is not a good idea...
        X_mb = get_batch(samples, batch_size, batch_idx + d + offset)
        X_mb = X_mb.reshape(batch_size,seq_length,1)
        Y_mb = get_batch(samples, batch_size, batch_idx + d + offset)
        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={CD:Y_mb, CG:Y_mb, X:X_mb, 
                                             Z: sample_Z(batch_size, seq_length, latent_dim)})
    return D_loss_curr


# Functions used to visualize generated data
def visualise_at_epoch(vis_sample, epoch, num_epochs):

    save_plot_sample(vis_sample, epoch, n_samples=6,
                     num_epochs=num_epochs) 
    return True

def save_plot_sample(samples, idx, n_samples=6, num_epochs=None, ncol=2):
    assert n_samples <= samples.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples.shape[1]
  
    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    for m in range(nrow):
        for n in range(ncol):
            # first column
            sample = samples[n*nrow + m, :, 0]
            axarr[m, n].plot(x_points, sample, color=col)
            axarr[m, n].set_ylim(-1, 1)
    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)
    plt.clf()
    plt.close()
    return

# --- train --- #

print('num_epoch\t D_loss_curr\t G_loss_curr\t time elapsed')  

d_loss = []
g_loss = []
for num_epoch in range(num_epochs):
    
    for batch_idx in range(0, int(len(samples)/batch_size) - (D_rounds + G_rounds), D_rounds + G_rounds): 
        
        if num_epoch % 2 == 0: 
            
            G_loss_curr = train_generator(batch_idx,0)
            D_loss_curr = train_discriminator(batch_idx, G_rounds)
            
        else: 
            
            D_loss_curr = train_discriminator(batch_idx, 0)
            G_loss_curr = train_generator(batch_idx, D_rounds)
        
        d_loss.append(D_loss_curr)
        g_loss.append(G_loss_curr)
        t = time() -t0
        
       
        print(num_epoch,'\t', D_loss_curr, '\t', G_loss_curr, '\t', t)
        
    # save synthetic data
    if num_epoch % 5 == 0:
    # generate synthetic dataset
        gen_samples = []
        for batch_idx in range(int(len(samples) / batch_size)):
            X_mb = get_batch(samples, batch_size, batch_idx)
            Y_mb = get_batch(samples, batch_size, batch_idx)
            z_ = sample_Z(batch_size, seq_length, latent_dim)
            gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_, CG:Y_mb})
            gen_samples.append(gen_samples_mb)
            print (batch_idx)

        gen_samples = np.vstack(gen_samples)



ax = pd.DataFrame(
    {
        'Generative Loss': g_loss,
        'Discriminative Loss': d_loss,
    }
).plot(title='Training loss', logy=True)
ax.set_xlabel("Training iterations")
ax.set_ylabel("Loss")


generated_data = np.transpose(gen_samples)
#plot the log-returns
gen_ind = 1 # change in function price as well
pd.DataFrame(generated_data[0,:,gen_ind]).plot()  # 1 is the index of the plotted sample out of the 1000 generated

# get the prices from the log returns
def price_gen(ind_gen_sample = 1): 
    
    daily_log_returns=generated_data[0,:,ind_gen_sample]
    
    price_list = [S] #initial price (today)

    for x in np.transpose(daily_log_returns): 

        price_list.append(price_list[-1]*np.exp(x))
        
    #price_list = np.asarray(price_list)
    
    return price_list

def price_real(): 
    
    daily_returns=np.random.normal(mu/T,vol/math.sqrt(T),T)+1
    
    price_list = [S]
 
    for x in daily_returns:
        
        price_list.append(price_list[-1]*x)
        
    return price_list


def prices_gen_data_frame(num_samples = 1): 
    df = pd.DataFrame([])
    
    for i in range(num_samples): 
        
        df[i] = price_real()
    
    return df

def prices_real_data_frame(num_samples = 1): 
    df = pd.DataFrame()
    
    for i in range(num_samples): 
        
        df[i] = price_real()
    
    return df

# Plot the price evolution of the generated sample
def plot_price_gen(num_gen_sample = 1): 
    
    for i in range(num_gen_sample): 
        daily_log_returns = generated_data[0,:,i]
        
        price_list = [S]
        
        for x in daily_log_returns: 
            price_list.append(price_list[-1]*np.exp(x))
        
        plt.plot(price_list)
        
    return plt.show()

# Plot one real evolution
def plot_price_real(num_gen_samples = 1): 
    
    for i in range(num_gen_samples):
        #create list of daily returns using random normal distribution
        daily_returns=np.random.normal(mu/T,vol/math.sqrt(T),T)+1
 
        #set starting price and create price series generated by above random daily returns
        price_list = [S]
 
        for x in daily_returns:
            price_list.append(price_list[-1]*x)
        
        #plot data from each individual run which we will plot at the end
        plt.plot(price_list)
        #pd.DataFrame(price_list).plot() #(if we want plots on separate figures)
    
    return plt.show() # return None (if we want plots on separate figures)


#Statistical tests on the final prices

def transform2darray(vector):
    v = []
    for i in range(len(vector)):
        v.append(vector[i][0])
    return v

num_sample = 50000
result_gen = prices_gen_data_frame(num_sample).iloc[[seq_length-1]]
result_gen = np.asarray(result_gen)
result_gen = np.transpose(result_gen)
result_gen = transform2darray(result_gen)
result_real = prices_real_data_frame(num_sample).iloc[[seq_length-1]]
result_real = np.asarray(result_real)
result_real = np.transpose(result_real)
result_real = transform2darray(result_real)
results = pd.DataFrame({'Real':result_real, 'Fake': result_gen})

# statistics summary
print(results.describe())
# boxplot
results.boxplot()

results.hist(bins=100)

# Komogorov-Smirnov (This is a two-sided test for the null hypothesis that 2 independent samples are drawn from the same continuous distribution.)
# calculate the significance
value, pvalue = ks_2samp(result_real, result_gen)
print("test statistic =", value, "  ", "pvalue = ", pvalue)
if pvalue > 0.05:
	print('Samples are likely drawn from the same distributions (not reject H0)')
else:
	print('Samples are likely drawn from different distributions (reject H0)')