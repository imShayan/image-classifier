import tensorflow as tf
#STEP1: LOADING INPUT DATASET
from tensorflow.examples.tutorials.mnist import input_data

mnist =  input_data.read_data_sets("/tmp/data",one_hot=True)
'''
WHAT IS ONE_HOT= TRUE  ?
ANS: HERE IN THIS MODEL WE ARE DOING A MULTI-CLASS CLASSIFICATION
SAY OUTPUT LABELS WILL BE LIKE :
0-->0
1-->1
.
.
9-->9
AFTER APPLYING ONE_HOT THINGS LOOK LIKE:
0=[0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0]
2=[0,0,1,0,0,0,0,0,0]
'''
# STEP 2: NOW WE TO CREATE A AND DEFINE THE HIDDEN LAYERS
# NO OF NODES IN EACH HIDDEN LAYER

no_nodes_hl1 =500
no_nodes_hl2 =500
no_nodes_hl3 =500

#MENTION NO OF CLASSES ON WHICH WE NEED WE NEED TO CLASSIFY IT CAN 2,3 IN THIS CASE 10
no_classes=10

#BATCH SIZE TELLS US --> HOW MANY INPUT DATA(IMAGES) WE ARE GOING TO PICK FROM DATASET
batch_size=100
#('FLOAT',[NONE ,784] MEANS A VARIABLE WHICH STORES FLOAT TYPE DATA OF HEIGHT=NONE
#WIDTH=784
x=tf.placeholder('float',[None, 784])
# Y IS THE LABEL OR OUTPUT WHICH DONT KNOW
y=tf.placeholder('float')

# NOW WE START TO DESIGN THE DEEP NEURAL NETWORK
# FUNCTION THAT TAKES DATA AS INPUT
def neural_network_model(data):
    #HIDDEN LAYER IS  DICTIONARY WHICH HAS WEIGTHS AND BAISES
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,no_nodes_hl1])),
                      'baises': tf.Variable(tf.random_normal([no_nodes_hl1]))}


    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([no_nodes_hl1,no_nodes_hl2])),
                      'baises': tf.Variable(tf.random_normal([no_nodes_hl2]))}


    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([no_nodes_hl2, no_nodes_hl3])),
                      'baises': tf.Variable(tf.random_normal([no_nodes_hl3]))}


    output_layer = {'weights': tf.Variable(tf.random_normal([no_nodes_hl3, no_classes])),
                      'baises': tf.Variable(tf.random_normal([no_classes]))}

    # BY FAR WE JUST DESIGNED THE NEURAL NETWORK BUT WE STILL NEED TO BUT THE MODEL :
    # MODEL ---> (INPUT* WEIGHTS) + BAISES
    # PASS THIS DATA THROUGH ACTIVATION FUNCTION

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']),hidden_1_layer['baises'])
    l1= tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']),hidden_2_layer['baises'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['baises'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['baises']

    return output

# HERE WE CONCLUDE TO BUILD THE MODEL
# WE ARE DONE WITH THE COMPUTING GRAPH

#NOW LETS START TRAINING OUR MODEL

def train_neural_network(x):
    #INVOKING THE MODEL TO MAKE PREDICTIONS
    prediction=neural_network_model(x)
    # WE NEED TO DEFINE A COST FUNCTION TO FIND DIFFFERENCE BETWEEN THE
    #ACTUAL OUTPUT AND THE PREDICTION

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    #NOW WE NEED TO OPTIMIZE THE  COST SO WE APPLY AN OPTIMIZER
                #   LEARNING RATE BY DEFAULT IS 0.001
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    #EPOCH = FEED FORWARD +BACKPROP

    no_epoch=5

    #WE OPEN THE SESSION

    with tf.Session() as sess:
        # WE INITIALIZE ALL THE VARIABLE
        sess.run(tf.initialize_all_variables())

        # LOOPING THROUGH EACH EPOCH
        for epoch in range(no_epoch):
            epoch_loss=0
            # _ IN FOR LOOP MEANS VARIABLE WE DONT CARE ABOUT THIS LOOP
            # SHOW THAT HOW MANY TIME THIS LOOP WILL RUN IF MY BATCH SIZE WAS 100
            # LIKE 60000/100 OR SOMETHING
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # TRAINING THE MODEL TO CLASSIFIY THE ON FIRST 100 IMAGES
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)

                # THIS LINE OF CODE IS OPTIMIZING THE LOSS OR COST ON DATA OF 100 IMAGES
                # FEED_DICT MEANS WE THE FEEDING THE DATA THROUGH DICTIONARY
                _,c = sess.run([optimizer , cost] , feed_dict= {x:epoch_x, y: epoch_y})

                epoch_loss+=c

            print('EPOCH',epoch ,'COMPLETED OUT OF',no_epoch, 'LOSS =' ,epoch_loss)

            
            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            # TF.CAST DOES THE TYPE CASTING OF DATA TYPE HERE FLOAT
            accuracy= tf.reduce_mean(tf.cast(correct ,'float'))

            print('Accuracy:', accuracy.eval({x:mnist.test.images ,y:mnist.test.labels}))


train_neural_network(x)



