import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
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
  print("Hello")


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    points_expanded = tf.expand_dims(X, 0)
    means_expanded = tf.expand_dims(MU, 1)    
    return tf.reduce_sum(tf.square(points_expanded - means_expanded), 2)


def buildGraph(input_data, cluster_size):
    input_x = tf.placeholder(tf.float32, shape=[None, input_data.shape[1]], name='input_x')
    centroids = tf.Variable(tf.random_normal([cluster_size, input_data.shape[1]], stddev=0.5))
    
    #Calculate the pair-distance matrix
    pair_distance = distanceFunc(input_x, centroids)
    
    #Calculate the loss
    loss = tf.reduce_sum(tf.reduce_min(pair_distance, 1))
    
    #Optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
    
    return input_x, centroids, pair_distance, loss, optimizer
    

def main(): 
    #loss values
    train_loss_list = []
    
    #Build the graph
    input_x, centroids, pair_distance, loss, optimizer = buildGraph(data, 3)
    
    with tf.Session() as sess:
    # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Loop over number of epochs
        for epoch in range(1000):
            feed_dict_train = {input_x: data}
            
            #Run the optimizer
            sess.run(optimizer, feed_dict=feed_dict_train)
            
            loss_value = sess.run(loss, feed_dict=feed_dict_train)
            train_loss_list.append(loss_value)
            print(loss_value)
     

    plt.figure(1)
    plt.plot(train_loss_list, c='b')
    plt.title("Cluster Size of 3")
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')          

if __name__ == "__main__":
    main()
    
    
