## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from setup_mnist import MNIST, MNISTModel
from l2_attack import CarliniL2

import matplotlib.pyplot as plt
import os




if __name__ == "__main__":

    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
                
        attack = CarliniL2(sess, model, batch_size=9, max_iterations=1000, confidence=0)
        



        # samples=1 means only 1 type of images in a class should be selected.
        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)
        # inputs has only 1 sample which is only 1 image and it is repeated 9 times.

        timestart = time.time()
        
        file_name='adv_attack_dataset.npz'
        if not os.path.exists(file_name):
            adv = attack.attack(inputs, targets)
            print('saving adversarial attack dataset...')
            np.savez(file_name, adv=adv)
            print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        print('loading adversarial attack dataset...')            


        f=np.load(file_name)
        adv=f['adv']

        timeend = time.time()

        adv_shape=[1,adv.shape[1],adv.shape[2],adv.shape[3]]
        input_shape=[1,inputs.shape[1],inputs.shape[2],inputs.shape[3]]
        for i in range(len(adv)):
            d=inputs[i].reshape((28,28))
            e=adv[i].reshape((28,28))
            plt.imshow(d)
            #plt.show()
            plt.imshow(e)
            #plt.show()        	
            
            print("Correct Classification:", model.model.predict_classes(inputs[i].reshape(input_shape)))            
            
            print("Classification:", model.model.predict_classes(adv[i].reshape(adv_shape)))            

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
