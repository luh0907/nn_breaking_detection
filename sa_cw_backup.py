# Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>
# All rights reserved.

import sys
import time
import tensorflow as tf
import numpy as np
import random

import imageio

from setup_cifar import CIFARModel, CIFAR
from setup_mnist import MNISTModel, MNIST

sys.path.append("../..")
from nn_robust_attacks.l2_attack import CarliniL2
from fast_gradient_sign import FGS

import keras
from keras import backend as K
#EHLEE
from keras.models import Model

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
from multiprocessing import Pool

BINARY_SEARCH_STEPS = 9  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 10000   # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2     # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1e-3     # the initial constant c to pick as a first guess

TARGET_CLASS = 9

class CarliniL2New:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST, extra_loss=None, debug=None, de=None, target_lab=None):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        """

        image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        shape = (batch_size,image_size,image_size,num_channels)
        
        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.origs = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)
        self.const2 = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_origs = tf.placeholder(tf.float32, shape)
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        self.assign_const2 = tf.placeholder(tf.float32, [batch_size])
        
        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        self.newimg = tf.tanh(modifier + self.timg)/2
        
        # prediction BEFORE-SOFTMAX of the model
        self.output = model.predict(self.newimg)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-tf.tanh(self.origs)/2),[1,2,3])
        
        # compute the probability of the label class versus the maximum other
        self.real = real = tf.reduce_sum((self.tlab)*self.output,1)
        self.other = other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab*10000),1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        if extra_loss != None:
            self.extra_loss = extra_loss(self.newimg, self.output)
        else:
            self.extra_loss = 0
        self.loss = self.loss1+self.loss2+self.const*tf.reduce_sum(self.extra_loss)

        if debug != None:
            self.debug = debug(self.newimg)
        else:
            self.debug = self.newimg
        self.de = de
        self.target_lab = target_lab
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.origs.assign(self.assign_origs))
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.const2.assign(self.assign_const2))
        
        self.init = tf.variables_initializer(var_list=[modifier]+new_vars)

    def attack(self, origs, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            print('tick',i)
            r.extend(self.attack_batch(origs[i:i+self.batch_size], 
                                       imgs[i:i+self.batch_size], 
                                       targets[i:i+self.batch_size]))
        return np.array(r)

    def attack_batch(self, origs, imgs, labs):
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                x[y] -= self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        imgs = np.arctanh(imgs*1.999999)
        origs = np.arctanh(origs*1.999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        CONST2 = np.ones(batch_size)*self.initial_const

        # the best l2, score, and image attack
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.zeros(imgs[0].shape)]*batch_size
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl2 = [1e10]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_origs: origs,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST,
                                       self.assign_const2: CONST2})
            
            print('set new const',CONST)
            prev = 1e20
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack 
                _, l, l2s, scores, nimg, extra = self.sess.run([self.train, self.loss, 
                                                         self.l2dist, self.output, 
                                                         self.newimg, self.extra_loss])
                #print(np.argmax(scores))
                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,*self.sess.run((self.loss,self.loss1,self.loss2,self.extra_loss)))
                    print(*self.sess.run((self.debug)))
                    print(self.de[self.target_lab].predict(nimg))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l

                # adjust the best result found so far
                for e,(l2,sc,ii) in enumerate(zip(l2s,scores,nimg)):
                    #print(extra.shape)
                    if l2 < bestl2[e] and compare(sc, np.argmax(batchlab[e])) and extra[e] <= 0:
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    #print(l2,o_bestl2[e],np.argmax(sc),np.argmax(batchlab[e]),
                    #      extra[e])
                    if l2 < o_bestl2[e] and compare(sc, np.argmax(batchlab[e])) and extra[e] <= 0:
                        #print('set')
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 10

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack

def pop(model):
    '''Removes a layer instance on top of the layer stack.
    This code is thanks to @joelthchao https://github.com/fchollet/keras/issues/2371#issuecomment-211734276
    '''
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    else:
        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False

    return model

# EHLEE
def pop_layer(model, layer_name):
    temp_model = Model(
            inputs = model.input,
            outputs = [model.get_layer(layer_name).output],
    )


    return temp_model

# EHLEE - from SA github
def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

class DensityEstimate:
    def __init__(self, sess, hidden, centers, image_size, num_channels, removed_cols, sigma=20):
        self.sess = sess
        #print("Center shape (before): ")
        #print(centers.shape)
        #centers = hidden.predict(centers).reshape((centers.shape[0],1,-1))
        self.centers_shape0 = centers.shape[0]
        #p = Pool(10)
        centers = hidden.predict(centers)
        #centers = np.array(p.map(_aggr_output, [centers[i] for i in range(centers.shape[0])]))
        centers = np.array([_aggr_output(centers[i]) for i in range(centers.shape[0])])
        '''
        removed_cols = []
        col_vectors = np.transpose(centers)
        for i in range(col_vectors.shape[0]):
            if(np.var(col_vectors[i]) < 1e-5 and i not in removed_cols):
                removed_cols.append(i)
        '''
        self.removed_cols = removed_cols
        centers = np.transpose(centers)
        centers = np.delete(centers, removed_cols, axis=0)
        self.gaussian_kde = gaussian_kde(centers)
        centers = np.transpose(centers)
        #print("Center shape (interm): ")
        #print(centers.shape)
        #centers = centers.reshape((5444, 1, -1))
        #print("Center shape (after): ")
        print(centers.shape)
        self.centers = centers

        self.sigma = sigma

        self.gaussian_means = tf.constant(centers)

        self.X = tf.placeholder(tf.float32, (None, image_size, image_size, num_channels))

        #hidden_res = hidden(self.X)[tf.newaxis,:,:]
        hidden_res = hidden(self.X)
        hidden_res = tf.stack([tf.reduce_mean(hidden_res[..., j]) for j in range(hidden_res.shape[-1])])
        hidden_res = hidden_res[tf.newaxis,:]
        hidden_res = tf.transpose(hidden_res)
        remained_cols = [item for item in range(hidden_res.shape[0]) if not item in removed_cols]
        hidden_res = tf.gather(hidden_res, remained_cols)
        hidden_res = tf.transpose(hidden_res)
        #print(hidden_res.shape)
        
        self.remained_cols = remained_cols

        self.dist = tf.reduce_sum(tf.reshape(tf.square(self.gaussian_means - hidden_res),(self.centers_shape0,1,-1)),axis=2)
        #self.dist = tf.reduce_sum(tf.square(self.gaussian_means - hidden_res),axis=1)

        self.Y = tf.reduce_mean(tf.exp(-self.dist/self.sigma),axis=0)
        self.hidden = hidden

    def make(self, X):
        #dist = tf.reduce_sum(tf.reshape(tf.square(self.gaussian_means - self.hidden(X)[tf.newaxis,:,:]),(self.centers_shape0,1,-1)),axis=2)
        #dist = tf.reduce_sum(tf.reshape(tf.square(self.gaussian_means - self.hidden(X)),(self.centers_shape0,1,-1)),axis=2)

        hidden_res = self.hidden(X)
        hidden_res = tf.stack([tf.reduce_mean(hidden_res[..., j]) for j in range(hidden_res.shape[-1])])
        hidden_res = hidden_res[tf.newaxis,:]
        hidden_res = tf.transpose(hidden_res)
        hidden_res = tf.gather(hidden_res, self.remained_cols)
        kde_tensor=tf.py_func(func=self.gaussian_kde.pdf, inp=[hidden_res], Tout=tf.float64)
        kde_tensor = tf.cast(kde_tensor, tf.float32)
        #self.gaussian_kde.pdf(hidden_res)
        hidden_res = tf.transpose(hidden_res)
        
        dist = tf.reduce_sum(tf.reshape(tf.square(self.gaussian_means - hidden_res),(self.centers_shape0,1,-1)),axis=2)
        #dist = tf.reduce_sum(tf.square(self.gaussian_means - hidden_res),axis=1)

        #return tf.reduce_mean(tf.exp(-dist/self.sigma),axis=0)
        return kde_tensor

    def predict(self, xs):
        #print(xs.shape)
        #print(self.gaussian_means.shape)

        hidden_res = self.hidden.predict(xs)[0]
        hidden_res = np.array(_aggr_output(hidden_res))
        #hidden_res = hidden_res[tf.newaxis,:]
        hidden_res = np.delete(hidden_res, self.removed_cols, axis=0)
        #hidden_res = np.transpose(hidden_res)
        #print(hidden_res.shape)

        return [np.asscalar(self.gaussian_kde.pdf(hidden_res))]
        #return self.sess.run(self.Y, {self.X: xs})

def estimate_density_full(model, de, data, labels=None):
    if labels is None:
        labels = model.model.predict(data)

    res = []
    for j in range(0,len(data),1):
        i = np.argmax(labels[j])
        probs = de[i].predict(data[j:j+1])
        res.extend(probs)
    return np.array(res)

def extra_loss(de, target_lab):
    def fn(img, out):
        return tf.nn.relu(-tf.log(de[target_lab].make(img))-DECONST)
    return fn

def debug_extra_loss(de, target_lab):
    def fn(img):
        return -tf.log(de[target_lab].make(img))
    return fn

def compute_optimal_sigma(sess, model, hidden_layer, data):
    sigma = tf.Variable(np.ones(1)*100,dtype=tf.float32)
    de = [DensityEstimate(sess, hidden_layer, data.train_data[np.argmax(data.train_labels,axis=1)==i], model.image_size, model.num_channels, sigma) for i in range(10)]
    #print(de[0].centers)
    #print(estimate_density(model, de, data.test_data))
    xs = []
    for const in np.arange(-1,0,.02):
        sess.run(sigma.assign(np.ones(1)*(10**const)))
        r = []
        for labA in range(10):
            #print(labA)
            for labB in range(10):
                subset = data.validation_data[np.argmax(data.validation_labels,axis=1)==labB,:,:,:]
                r.append(np.mean(np.log(1e-30+de[labA].predict(subset))))
        r = np.array(r).reshape((10,10))
        diag = np.mean(r[np.arange(10),np.arange(10)])
        r[np.arange(10),np.arange(10)] = 0
        rest = np.mean(r)
        value = diag-rest
        xs.append(value)
    print(xs)
    plt.plot(np.arange(-1,0,.02),xs)
    plt.xlabel('sigma')
    plt.ylabel('Log liklihood difference')
   
    pp = PdfPages('/tmp/aaa.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()

    #plt.show()
    
    exit(0)
    
def get_removed_cols(hidden, centers):
    removed_cols = []
    centers = hidden.predict(centers)
    centers = np.array([_aggr_output(centers[i]) for i in range(centers.shape[0])])
    col_vectors = np.transpose(centers)
    for i in range(col_vectors.shape[0]):
        if(np.var(col_vectors[i]) < 1e-5 and i not in removed_cols):
            removed_cols.append(i)
    return removed_cols
    
def run_kde(Data, Model, path):
    global DECONST
    sess = K.get_session()
    K.set_learning_phase(False)
    data, model = Data(), Model(path)

    model2 = Model(path)

    # TODO: hidden_layer -> selected layer
    layer_name = "activation_7"
    hidden_layer = pop_layer(model2.model, layer_name)
    #hidden_layer = pop(model2.model) # once to remove dense(10)
    #hidden_layer = pop(hidden_layer) # once to remove ReLU

    #compute_optimal_sigma(sess, model, hidden_layer, data)
    #MNIST SIGMA: 20
    #bandwidth: 0.864
    
    removed_cols = []
    for i in range(10):
        removed_cols.extend(get_removed_cols(hidden_layer, data.train_data[np.argmax(data.train_labels,axis=1)==i]))
    removed_cols = list(set(removed_cols))

    de = [DensityEstimate(sess, hidden_layer, data.train_data[np.argmax(data.train_labels,axis=1)==i], model.image_size, model.num_channels, removed_cols) for i in range(10)]
    de2 = [DensityEstimate(sess, hidden_layer, data.train_data[np.argmax(data.train_labels,axis=1)==i][:100], model.image_size, model.num_channels, removed_cols) for i in range(10)]

    p = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))

    #print(np.log(de[0].predict(data.test_data[:10])))
    #print(sess.run(rmodel.predict(p)[1], {p: data.test_data[:10]}))
    #exit(0)

    N = 9
    #print(model.model.predict(data.train_data[:N]))
    #print(hidden_layer.predict(data.train_data[:N]))

    adv_candid = []
    jumped = False
    adv_labels = np.zeros((9,10))
    for i in range(0,10):
        if i == TARGET_CLASS:
            jumped = True
            continue
        adv_candid.extend(data.test_data[np.argmax(data.test_labels,axis=1)==i][:1])
        if jumped:
            adv_labels[i-1][TARGET_CLASS] = 1
        else:
            adv_labels[i][TARGET_CLASS] = 1

    adv_candid = np.array(adv_candid)

    #for i in range(10):
    #    for j in range(N):
    #        print(de[i].predict(data.train_data[j:j+1])) # N
    
    #start_density = estimate_density_full(model, de, data.test_data[M:M+N])+1e-30
    start_density = estimate_density_full(model, de, adv_candid)+1e-30
    #start_density = estimate_density_full(model, de, data.train_data, labels=data.train_labels)+1e-30
    #print("starting density", np.log(start_density))

    DECONST = -np.log(start_density)
    DECONST = np.median(DECONST)
    #DECONST = -90.0

    l = np.zeros((N,10))
    #l[np.arange(N),np.random.random_integers(0,9,N)] = 1
    for i in range(N):
        r = np.random.random_integers(0,9)
        while r == np.argmax(data.test_labels[i]):
            r = np.random.random_integers(0,9)
        l[i,r] = 1

    l = adv_labels
    print(l)
    attack1 = CarliniL2(sess, model, batch_size=1, max_iterations=3000,
                       binary_search_steps=3, initial_const=1.0, learning_rate=1e-1,
                       targeted=True)
    attack2 = CarliniL2New(sess, model, batch_size=1, max_iterations=10000,
                           binary_search_steps=5, initial_const=1.0, learning_rate=1e-2,
                           targeted=True, extra_loss=extra_loss(de2, TARGET_CLASS), debug=debug_extra_loss(de2, np.argmax(l)), de=de2, target_lab=TARGET_CLASS)
    #l = data.test_labels[:N]
    #l = np.zeros((N,10))
    #l[np.arange(N),1] = 1
    print("RUN PHASE 1")
    #adv = attack1.attack(data.test_data[M:M+N], l)
    adv = attack1.attack(adv_candid, l)
    #print('mean distortion',np.mean(np.sum((adv-data.test_data[M:M+N])**2,axis=(1,2,3))**.5))
    print('mean distortion', np.mean(np.sum((adv-adv_candid)**2, axis=(1,2,3))**.5))

    print("RUN PHASE 2")
    #adv = attack2.attack(data.test_data[M:M+N], adv, l)
    adv = attack2.attack(adv_candid, adv, l)

    #np.save("/tmp/q"+str(M),adv)
    np.save("./adv/adv_mnist_cnw_target_{}".format(TARGET_CLASS), adv)
    #adv = np.load("/tmp/qq.npy")

    #print('labels',np.mean(np.argmax(sess.run(model.predict(p), {p: adv}),axis=1)==l))
    print('labels')
    print(np.argmax(l, axis=1))
    print(np.argmax(sess.run(model.predict(p), {p: adv}), axis=1))
    print(np.argmax(model.model.predict(adv), axis=1))

    #print('mean distortion',np.mean(np.sum((adv-data.test_data[M:M+N])**2,axis=(1,2,3))**.5))
    print('mean distortion', np.mean(np.sum((adv-adv_candid)**2, axis=(1,2,3))**.5))
    
    #a = estimate_density_full(model, de, data.test_data[M:M+N])+1e-30
    a = estimate_density_full(model, de, adv_candid)+1e-30
    b = estimate_density_full(model, de, adv)+1e-30

    #print(data.test_data.shape)
    #print(adv.shape)

    show(adv)

    print('de of test', np.mean(np.log(a)))
    print('de of adv', np.mean(np.log(b)))

    print('better ratio', np.mean(np.array(a)>np.array(b)))
    exit(0)

    #density = gaussian_kde(np.array(np.log(a))-np.array(np.log(b)))
    #density_a = gaussian_kde(np.log(a))
    #density_b = gaussian_kde(np.log(b))

    xs = np.linspace(-25,25,200)
    
    fig = plt.figure(figsize=(4,3))
    fig.subplots_adjust(bottom=0.17,left=.15, right=.85)
    
    plt.xlabel('log(KDE(valid))-log(KDE(adversarial))')
    plt.ylabel('Occurrances')
    
    #plt.hist(np.log(a),100)
    #plt.hist(np.log(b),100)
    plt.hist(np.log(a)-np.log(b),100)
    #plt.hist(np.array(np.log(a))-np.array(np.log(b)),100)
    #a = plt.plot(xs,density_a(xs), 'r--',color='blue', label='Valid')
    #b = plt.plot(xs,density_b(xs), color='red', label='Adversarial')
    #plt.plot(xs,density(xs))
    
    #plt.legend(handles=[a[0], b[0]])
    
    pp = PdfPages('/tmp/a.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()
    #plt.show()

def show(img):
    jumped = False
    for i in range(10):
        if i == TARGET_CLASS:
            jumped = True
            continue
        if jumped:
            imageio.imwrite("./adv/adv_result_{}_to_{}.png".format(i,TARGET_CLASS), img[i-1].reshape(28,28))
        else:
            imageio.imwrite("./adv/adv_result_{}_to_{}.png".format(i,TARGET_CLASS), img[i].reshape(28,28))

    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    print
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
        #print([x for x in img[i*28:i*28+28]])

#M = int(sys.argv[1])
M = 18
run_kde(MNIST, MNISTModel, "models/mnist")
#run_kde(CIFAR, CIFARModel, "models/cifar")
