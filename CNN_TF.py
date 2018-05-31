import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials import mnist
from scipy.io import loadmat
data = input_data.read_data_sets("/tmp/data/", one_hot = True)
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
node1 = 500
node2 = 500
node3 = 500

classes = 10
batch_size = 50


"""data = loadmat('ex4data1.mat')


YY = data['y']
X = data['X']

encoder = LabelEncoder()
encoder.fit(YY)
temp = encoder.transform(YY)
Y = np_utils.to_categorical(temp)

"""
#Y = Y.astype(float)
#X = X.astype(float)

X = tf.placeholder('float',[None , 784])
Y = tf.placeholder('float',)
keeprate = 0.8
nclasses = 10
def conv(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



def CNN_model(x):

    weights = {'W1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W2':tf.Variable(tf.random_normal([5,5,32,64])),
               'Final':tf.Variable(tf.random_normal([64*7*7,1024])),
                'output':tf.Variable(tf.random_normal([1024,nclasses]))
               }


    biases = {'b1':tf.Variable(tf.random_normal([32])),
               'b2':tf.Variable(tf.random_normal([64])),
               'bfinal':tf.Variable(tf.random_normal([1024])),
               'output':tf.Variable(tf.random_normal([nclasses]))}

    x = tf.reshape(x,shape=[-1,28,28,1])

    conv1 = tf.nn.relu(conv(x,weights['W1']+biases['b1']))
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv(conv1,weights['W2']+biases['b2']))
    conv2 = maxpool2d(conv2)

    final = tf.reshape(conv2,shape=[-1,7*7*64])



    final = tf.nn.relu(tf.matmul(final,weights['Final'])+biases['bfinal'])
    fc = tf.nn.dropout(final, keeprate)
    output = tf.matmul(fc,weights['output'])+biases['output']


    return output


def train(x):
    pred = CNN_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer= tf.train.AdamOptimizer().minimize(cost)

    n_epoch = 7

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(n_epoch):
            loss_epoch = 0
            for _ in range(int(data.train.num_examples/batch_size)):
                xx,yy = data.train.next_batch(batch_size)
                _,c = sess.run([optimizer,cost],feed_dict={x:xx,Y:yy})
                loss_epoch += c

            print('Epoch', epoch, 'completed out of', n_epoch, 'loss:',loss_epoch)
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print("Acc : ", accuracy.eval({x:data.test.images ,Y:data.test.labels}))


train(X)
