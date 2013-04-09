import numpy as np
from pylearn2.utils import serial
import theano.tensor as T
from theano import function

DATA_DIR = '/home/nico/Code/kaggle/GenderWrite/'

# load configuration file and train model
train_obj = serial.load_train_file(DATA_DIR+'gwmaxout.yaml'
train_obj.main_loop()


# generate model output
def get_output(model, data):
    test.X = test.X.astype('float32')
    test.y = test.y.astype('float32')

    Xb = model.get_input_space().make_batch_theano()
    Xb.name = 'Xb'
    yb = model.get_output_space().make_batch_theano()
    yb.name = 'yb'
    
    ymf = model.fprop(Xb)
    ymf.name = 'ymf'
    
    data = tdata.get_topological_view()
    
    yl = T.argmax(yb,axis=1)
    
    mf1acc = 1.-T.neq(yl , T.argmax(ymf,axis=1)).mean()
    
    batch_acc = function([Xb,yb],[mf1acc])

def accs():
    mf1_accs = []
    assert isinstance(test.X.shape[0], int)
    assert isinstance(batch_size, int)
    for i in xrange(test.X.shape[0]/batch_size):
        print i
        x_arg = test.X[i*batch_size:(i+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        mf1_accs.append( batch_acc(x_arg,
            test.y[i*batch_size:(i+1)*batch_size,:])[0])
    return sum(mf1_accs) / float(len(mf1_accs))
result = accs()
print 1. - result

DATA_DIR+'gw_preprocessed_test.pkl'

# reshape model output so that shape[0] = no. of pages

# save
np.save(DATA_DIR+'maxout_train.npy', trainout)
np.save(DATA_DIR+'maxout_test.npy', testout)
