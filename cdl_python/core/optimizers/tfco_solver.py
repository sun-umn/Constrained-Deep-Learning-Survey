import tensorflow.compat.v2 as tf
from six.moves import xrange
import tensorflow_constrained_optimization as tfco
import copy
import numpy as np
import time

class problem(tfco.ConstrainedMinimizationProblem):

    def __init__(self, model, data, threshold,obj_loss,constr_loss):
        self._model = model
        self._threshold = threshold # precision lower bound
        self._obj_loss = obj_loss
        self._constr_loss = constr_loss


        self._train_X_0 = tf.constant(data['train_X_0'], dtype=tf.float32)
        self._train_y_0 = tf.constant(data['train_y_0'], dtype=tf.float32)
        self._train_X_1 = tf.constant(data['train_X_1'], dtype=tf.float32)
        self._train_y_1 = tf.constant(data['train_y_1'], dtype=tf.float32)
        self._test_X_0 = tf.constant(data['test_X_0'], dtype=tf.float32)
        self._test_y_0 = tf.constant(data['test_y_0'], dtype=tf.float32)
        self._test_X_1 = tf.constant(data['test_X_1'], dtype=tf.float32)
        self._test_y_1 = tf.constant(data['test_y_1'], dtype=tf.float32)
        print(self._train_X_0.shape)


    @property
    def num_constraints(self):
        return 2

    def objective(self):
        pred0 =  self._model(self._train_X_0)
        pred1 = self._model(self._train_X_1)
        num0 = self._train_X_0.shape[0]
        num1 = self._train_X_1.shape[0]
        f = (self._obj_loss(pred0,self._train_y_0) \
             + self._obj_loss(pred1,self._train_y_1))/(num0 + num1)
        return f

    def constraints(self):
        pred0 =  self._model(self._train_X_0)
        pred1 = self._model(self._train_X_1)
        num0 = self._train_X_0.shape[0]
        num1 = self._train_X_1.shape[0]
        diff = self._constr_loss(pred0,self._train_y_0) / num0 \
             - self._constr_loss(pred1,self._train_y_1)/ num1
        #print(tf.multiply(pos_s, self._alpha-1))
        cons_tensors0 = diff - self._threshold
        cons_tensors1 = -diff - self._threshold
        print([cons_tensors0,cons_tensors1])
        return [cons_tensors0,cons_tensors1]
    
    def evaluation(self):
        pred0 =  self._model(self._train_X_0)
        pred1 = self._model(self._train_X_1)
        num0 = self._train_X_0.shape[0]
        num1 = self._train_X_1.shape[0]
        train_y_0 = tf.cast(self._train_y_0,tf.int32)
        train_y_1 = tf.cast(self._train_y_1,tf.int32)
        test_y_0 = tf.cast(self._test_y_0,tf.int32)
        test_y_1 = tf.cast(self._test_y_1,tf.int32)
        f = ((self._obj_loss(pred0,self._train_y_0) \
             + self._obj_loss(pred1,self._train_y_1))/(num0 + num1)).numpy()
        diff = np.abs((self._constr_loss(pred0,self._train_y_0) / num0 \
             - self._constr_loss(pred1,self._train_y_1)/ num1).numpy())
        #print(tf.multiply(pos_s, self._alpha-1))
        cons0 = diff - self._threshold
        cons1 = -diff - self._threshold
        # accuracy
        pred0 = tf.cast(tf.greater_equal(pred0, 0.5), tf.int32)
        pred1 = tf.cast(tf.greater_equal(pred1, 0.5), tf.int32)
        # Calculate the accuracy
        accuracy = ((tf.reduce_sum(tf.cast(tf.equal(pred0, train_y_0), tf.float32))\
            + tf.reduce_sum(tf.cast(tf.equal(pred1, train_y_1), tf.float32)))/(num0 + num1))\
            .numpy()
        
        #========== test ========
        pred0 =  self._model(self._test_X_0)
        pred1 = self._model(self._test_X_1)
        num0 = self._test_X_0.shape[0]
        num1 = self._test_X_1.shape[0]
        ft = ((self._obj_loss(pred0,self._test_y_0) \
             + self._obj_loss(pred1,self._test_y_1))/(num0 + num1)).numpy()
        difft = np.abs((self._constr_loss(pred0,self._test_y_0) / num0 \
             - self._constr_loss(pred1,self._test_y_1)/ num1).numpy())
        #print(tf.multiply(pos_s, self._alpha-1))
        cons0t = difft - self._threshold
        cons1t = -difft - self._threshold
        # accuracy
        pred0 = tf.cast(tf.greater_equal(pred0, 0.5), tf.int32)
        pred1 = tf.cast(tf.greater_equal(pred1, 0.5), tf.int32)
        # Calculate the accuracy
        accuracyt = ((tf.reduce_sum(tf.cast(tf.equal(pred0, test_y_0), tf.float32))\
            + tf.reduce_sum(tf.cast(tf.equal(pred1, test_y_1), tf.float32)))/(num0 + num1))\
            .numpy()
        
        return f,[cons0,cons1],diff,accuracy,\
            ft,[cons0t,cons1t],difft,accuracyt


    
class TFCO_problem():
    def __init__(
        self,
        cfg,
        data,
        device,
        model,
        fn = ''
        ): # dataset is in
        self.cfg = cfg
        obj_loss = self.cfg.get('OPTIMIZER','OBJ_LOSS')
        constr_loss = self.cfg.get('OPTIMIZER','CONSTR_LOSS')
        if obj_loss == 'BCE':
            obj_loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.SUM)
        if constr_loss == 'BCE': 
            constr_loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.SUM)
        threshold = self.cfg.getfloat('EXP','THRESHOLD')
        self.problem = problem(model,data,threshold,obj_loss,constr_loss)
        self.fn = fn
        self.model = model

    
    def saveLog(self):
        log = copy.deepcopy(self.output)
        for key in log:
            log[key] = np.array(log[key])
        np.savez(self.fn,**log)

    def train(self):
        self.output = {
            'epoch':0,
            'f':[],
            'f_test':[],
            'ci':[],
            'ci_test':[],
            'diff':[],
            'diff_test':[],
            'time':[],

            'accuracy':[],
            'accuracy_test':[]
        }
        
        #dual_scale: optional float defaulting to 1, a multiplicative scaling factor
        #applied to gradients w.r.t. the Lagrange multipliers.
        loss_fn, update_ops_fn, multipliers = tfco.create_lagrangian_loss(
            self.problem
            )
        
        optimizer = tf.keras.optimizers.AdamW()#Adagrad() #learning_rate=0.01
        var_list = (list(self.problem.trainable_variables)+ self.model.trainable_weights + [multipliers])
        start_time = time.time()
        maxit = int(self.cfg.getfloat('OPTIMIZER','MAX_ITER'))
        self.print_summary_every = self.cfg.getint('OPTIMIZER','PRINT_SUMMARY_EVERY')
        for ii in xrange(maxit):
            update_ops_fn()
            optimizer.minimize(loss_fn, var_list=var_list)
            print(multipliers)
            f,cons,diff,accuracy,\
            ft,const,difft,accuracyt = self.problem.evaluation()
            self.output['f'].append(f)
            self.output['f_test'].append(ft)
            self.output['ci'].append(cons)
            self.output['ci_test'].append(const)
            self.output['diff'].append(diff)
            self.output['diff_test'].append(difft)
            self.output['accuracy'].append(accuracy)
            self.output['accuracy_test'].append(accuracyt)

            if self.output['epoch'] % self.print_summary_every == 1:
                for key,value in self.output.items():
                    if isinstance(value, list) and len(value) > 0:  # Check if it's a non-empty list
                        print(f'>>> {key}: {value[-1]}')
                    else:
                        print(f'>>> {key}: {value}')

            self.output['epoch'] += 1
            self.output['time'].append(time.time()-start_time)

            self.saveLog()

            
        

