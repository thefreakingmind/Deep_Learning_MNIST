import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
#No of Nodes in the Neural Network
nodes_hl1 =  400
nodes_hl2 = 400
nodes_hl3 = 400

n_classes = 10
batch_size = 100

#Placeholders
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#Defining our Neural Neural Network
def NeuralNetwork(data):
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, nodes_hl1])),
					'biases':tf.Variable(tf.random_normal(nodes_hl1))}

	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
						'biases':tf.Variable(tf.random_normal(nodes_hl2))}


	hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([nodes_hl2, nodes_hl3])),
						'biases': tf.Variable(tf.random_normal(nodes_hl3))}



	output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl3, nodeode_hl4])),
						'biases': tf.Variable(tf.random_normal(n_classes))}



    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights'],hidden_1_layer['biases']))
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights'], hidden_layer_1['biases']))
    l3 = tf.nn.relu(l3)
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

#Defining Functuion for Training Neural Network
def TrainingNeualNetwork():
	prediction = TrainingNeualNetwork(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)


	total_epochs = 20
	with tf.Session() as sess:
		sess.run(tf.global_variable_initilizer())

		for epochs in range(total_epochs):
			epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

TrainingNeuralNetwork(x)


