import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

is_valid = False

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    points_expanded = tf.expand_dims(X, 1)
    means_expanded = tf.expand_dims(MU, 0)    
    return tf.reduce_sum(tf.square(points_expanded - means_expanded), 2)

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K
    
    # TODO
    
    pair_distance = distanceFunc(X, mu)
    d = X.shape[1].value
    sigma2 = tf.squeeze(tf.exp(sigma))
    
    return -(1/2) * d * tf.log(2*np.pi*sigma2) - (pair_distance/(2*sigma2))
       

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    log_pi = tf.squeeze(log_pi)
    return log_PDF + log_pi - hlp.reduce_logsumexp(log_PDF + log_pi, 0, keep_dims=True)
    
    
def calculate_loss(X, mu, sigma, log_pi):
    # Inputs
    # X: N x D
    # mu: K x D
    # sigma: K x 1
    # unnormalized_pi: K x 1
    # loss: scalar

    log_PDF = log_GaussPDF(X, mu, sigma)
    log_pi = tf.squeeze(log_pi)

    return - tf.reduce_mean(hlp.reduce_logsumexp(log_PDF + log_pi, 1, keep_dims=True))    
    
    
def buildGraph(input_data, cluster_size):
    input_x = tf.placeholder(tf.float32, shape=[None, input_data.shape[1]], name='input_x')
    mu = tf.Variable(tf.random_normal([cluster_size, input_data.shape[1]]))
    sigma = tf.Variable(tf.random_normal([cluster_size, 1]))
    pi = tf.Variable(tf.random_normal([cluster_size, 1]))
    
    log_PDF = log_GaussPDF(input_x, mu, sigma)
    log_pi = hlp.logsoftmax(pi)

    #num_data = tf.placeholder(tf.float32)
    
    #Calculate the loss
    loss = calculate_loss(input_x, mu, sigma, log_pi)
    
    #Calculate Prediction
    prediction = tf.argmax(log_posterior(log_PDF, log_pi), 1)
    
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0075, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)

    return input_x, mu, sigma, pi, loss, prediction, optimizer


def main(): 
    #loss values
    train_loss_list = []
    
    #Build the graph
    input_x, mu, sigma, pi, loss, prediction, optimizer = buildGraph(data, 3)
    
    with tf.Session() as sess:
    # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Loop over number of epochs
        for epoch in range(1000):
            feed_dict_train = {input_x: data}
            
            #Run the optimizer
            sess.run(optimizer, feed_dict=feed_dict_train)
            
            train_loss, train_prediction = sess.run([loss, prediction], feed_dict=feed_dict_train)
            #print(train_prediction)
            train_loss_list.append(train_loss)
            print(train_loss)
     

    plt.figure(1)
    plt.plot(train_loss_list, c='b')
    plt.title("Cluster Size of 3")
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')  

    #plot_scatter(3, data, label= train_prediction, centers= centroids)       

if __name__ == "__main__":
    main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    