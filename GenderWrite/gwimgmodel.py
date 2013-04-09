import numpy as np
from pylearn2.utils import serial
from theano import function

CODE_DIR = '/home/nico/Code/kaggle/GenderWrite/'
DATA_DIR = '/home/nico/datasets/Kaggle/GenderWrite/'

# load configuration file and train model
train_obj = serial.load_train_file(CODE_DIR+'gwconv.yaml')

# layer-by-layer training
#layers = train_obj.model.layers[:]
## first remove all but first and last layers and train first layer
#for ii in range(len(layers)-2):
#    train_obj.model.layers.pop(1)

train_obj.main_loop()

# now add layers and re-train
#for ii in range(len(layers)-2):
#    train_obj.model.layers.insert(1+ii,layers[1+ii])
#    train_obj.main_loop()

# generate model output
def get_output(model, tdata, layerindex=-1, batch_size=100):
    Xb = model.get_input_space().make_batch_theano()
    ymf = model.fprop(Xb, return_all=True)
    
    data = tdata.get_topological_view()
    
    propagate = function([Xb],ymf)
    
    output = []
    for ii in xrange(int(data.shape[0]/batch_size)):
        seldata = data[ii*batch_size:(ii+1)*batch_size,:]
        output.append(propagate(seldata)[layerindex])
    
    output = np.reshape(output,[data.shape[0],-1])
    
    return output

trainset = serial.load(DATA_DIR+'gw_preprocessed_tottrain.pkl')
testset = serial.load(DATA_DIR+'gw_preprocessed_test.pkl')

model = train_obj.model
outtrainset = get_output(model,trainset,-1)
outtestset = get_output(model,testset,-1)
# save test output as submission
np.savetxt(DATA_DIR+'model_comb.csv', outtestset[:,0], delimiter=",")

# construct data sets with model output
outtrainset = get_output(model,trainset,-2)
outtestset = get_output(model,testset,-2)
        
# reshape model output so that shape[0] = no. of pages


# save
np.save(DATA_DIR+'maxout_train', outtrainset)
np.save(DATA_DIR+'maxout_test', outtestset)
