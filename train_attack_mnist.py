
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython
import os
import time

import tensorflow as tf

import experiments
from all_CNN_c import All_CNN_C
from all_CNN_attack import All_CNN_C_Attack
import matplotlib.pyplot as plt
from load_mnist import load_small_mnist, load_mnist
from l2_attack import CarliniL2,generate_data

tf.random.set_random_seed(10)    
data_sets = load_small_mnist('data')    

num_classes = 10
input_side = 28
image_size = 28
input_channels = 1
num_channels=input_channels
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001 
decay_epochs = [10000, 20000]
hidden1_units = 8
hidden2_units = 8
hidden3_units = 8
conv_patch_size = 3
keep_probs = [1.0, 1.0]
train_dir='output'
model_name='mnist_small_all_cnn_c'

model = All_CNN_C(
    input_side=input_side, 
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir=train_dir, 
    log_dir='log',
    model_name=model_name)

num_steps = 120

# run_phase='all'
# if run_phase=='all':
model.train(
    num_steps=num_steps, 
    iter_to_switch_to_batch=10000000,
    iter_to_switch_to_sgd=10000000)

checkpoint_file=model.get_checkpoint()
iter_to_load = num_steps - 1

# if run_phase=='all':
#     known_indices_to_remove=[]
# else:
#     f=np.load('output/my_work2_mnist_small_all_cnn_c_iter-500k_retraining-100.npz')
#     known_indices_to_remove=f['indices_to_remove']


tf.reset_default_graph()
tf_graph2=tf.Graph()

with tf_graph2.as_default() as g:
    model_to_be_attacked = All_CNN_C_Attack(
        image_size=image_size, 
        num_channels=num_channels,
        num_classes=num_classes,
        conv_patch_size=conv_patch_size,
        hidden1_units=hidden1_units, 
        hidden2_units=hidden2_units,
        hidden3_units=hidden3_units,
        weight_decay=weight_decay)
    logits=model_to_be_attacked.load_model()
    saver=tf.train.Saver()
    acc=model_to_be_attacked.get_accuracy_op()



with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    checkpoint_to_load = "%s-%s" % (checkpoint_file, iter_to_load) 
    saver.restore(sess, checkpoint_to_load)
    print("Check if accuracy is the same as the saved model:")
    print(sess.run(logits,feed_dict={model_to_be_attacked.input_placeholder:data_sets.test.x[10:12],
                            model_to_be_attacked.labels_placeholder:data_sets.test.labels[10:12]}).shape)

    exit()
    attack = CarliniL2(sess,model_to_be_attacked,logits, batch_size=9, max_iterations=1000, confidence=0)

    inputs_to_attack, targets_to_attack = generate_data(data_sets, samples=1, targeted=True,start=0, inception=False)
    inputs_to_attack=inputs_to_attack.reshape(inputs_to_attack.shape[0], 28, 28, 1)

    timestart = time.time()

    adv_name='adv_attack_dataset.npz'
    if not os.path.exists(adv_name):
        adv = attack.attack(inputs_to_attack, targets_to_attack)
        print('saving adversarial attack dataset...')
        np.savez(adv_name, adv=adv)
        timeend = time.time()
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

    print('loading adversarial attack dataset...')            


    f=np.load(adv_name)
    adv=f['adv']
    print(adv.shape)


    adv_shape=[1,adv.shape[1],adv.shape[2],adv.shape[3]]
    input_shape=[1,inputs_to_attack.shape[1],inputs_to_attack.shape[2],inputs_to_attack.shape[3]]
    print(attack.predict_classes(inputs_to_attack[0].reshape(input_shape)))
    print(attack.predict_classes(adv[0].reshape(adv_shape)))

exit()
for i in range(len(adv)):
    d=inputs_to_attack[i].reshape((28,28))
    e=adv[i].reshape((28,28))
    plt.imshow(d)
    #plt.show()
    plt.imshow(e)
    #plt.show()         
    # print("Valid:")
    # show(inputs[i])
    # print("Adversarial:")
    # show(adv[i])

    print("Correct Classification:", attack.predict_classes(inputs_to_attack[i].reshape(input_shape)))            
    
    print("Classification:", attack.predict_classes(adv[i].reshape(adv_shape)))            

    print("Total distortion:", np.sum((adv[i]-inputs_to_attack[i])**2)**.5)
exit()

actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
    model, 
    test_idx=test_idx, 
    iter_to_load=iter_to_load, 
    num_to_remove=100,
    num_steps=30000, 
    remove_type='maxinf',
    known_indices_to_remove=known_indices_to_remove,
    force_refresh=True)

filename="my_work2_numSteps"+str(num_steps)+"_"+run_phase+".txt"
np.savetxt(filename, np.c_[actual_loss_diffs,predicted_loss_diffs],fmt ='%f6')

if run_phase=="all":
    np.savez(
        'output/my_work2_mnist_small_all_cnn_c_iter-500k_retraining-100.npz', 
        actual_loss_diffs=actual_loss_diffs, 
        predicted_loss_diffs=predicted_loss_diffs, 
        indices_to_remove=indices_to_remove
        )
