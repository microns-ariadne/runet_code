import keras
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import Adam
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np

from prepare_ecs import *
from synapse_detection import *
from models import *
import sys


from scipy.sparse import coo_matrix
import scipy.ndimage.measurements as measurements
import scipy.sparse as sparse
import scipy.ndimage as ndimage

from cremi import Volume
from cremi.io import CremiFile
from cremi.evaluation import Clefts

### Variables ###
learn_rate = 0.00001
momentum = 0.99
threshold = 200
valid_name = 'ecs_train'
models_dir = '/n/coxfs01/eric_wu/convlstm/models/'
models_check_dir = '/n/coxfs01/eric_wu/convlstm/checkpoints/'
#weight_filename = 'runet_edt_1000_2017-07-08-19:43_weights.h5'
weight_filename = ''
print "Custom weight?", weight_filename
doPrediction = False
doDilation = False
bufferSize = 0
modelType = "runet"
modelClass = getModelClass(modelType)
#modelClass.setFilename("runet_halfres")
(patchZ, patchZ_out, patchSize, patchSize_out) = modelClass.getPatchSizes()
#modelClass.setPatchSizes((patchZ, patchZ_out, 512, 512))
#patchSize = 512
#patchSize_out = 322
filename = modelClass.getFilename()
print (patchZ, patchZ_out, patchSize, patchSize_out)
cropSize = (patchSize-patchSize_out)/2
csZ = (patchZ-patchZ_out)/2

def normalize(img):
    if np.max(img) == np.min(img): return img
    return (img-np.min(img))/(np.max(img)-np.min(img))

def weighted_mse(y_true, y_pred):
    epsilon=0.00001
    y_pred = K.clip(y_pred,epsilon, 1-epsilon)
    # per batch positive fraction, negative fraction (0.5 = ignore)
    pos_mask = K.cast(y_true > 0.75, 'float32')
    neg_mask = K.cast(y_true < 0.25, 'float32')
    num_pixels = K.cast(K.prod(K.shape(y_true)[1:]), 'float32')
    pos_fracs = K.clip((K.sum(pos_mask)/num_pixels),0.01, 0.99)
    neg_fracs = K.clip((K.sum(neg_mask) /num_pixels),0.01, 0.99)

    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = 1.0 / (2 * pos_fracs)
    neg_weight = 1.0 / (2 * neg_fracs)

    per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights
    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error)
    return K.mean(batch_weighted_mse)

def unet_crossentropy_loss_sampled(y_true, y_pred):
    # weighted version of pixel-wise crossentropy loss function
    alpha = 0.6
    epsilon = 1.0e-5
    y_pred_clipped = T.flatten(T.clip(y_pred, epsilon, 1.0 - epsilon)) # get rid of values very close to 0 or 1
    y_true = T.flatten(y_true)
    # this seems to work
    # it is super ugly though and I am sure there is a better way to do it
    # but I am struggling with theano to cooperate
    # filter the right indices
    indPos = T.nonzero(y_true)[0]  # no idea why this is a tuple
    indNeg = T.nonzero(1 - y_true)[0]
    # shuffle
    n = indPos.shape[0]
    indPos = indPos[srng.permutation(n=n)]
    n = indNeg.shape[0]
    indNeg = indNeg[srng.permutation(n=n)]

    # take equal number of samples depending on which class has less
    n_samples = T.cast(T.min([T.sum(y_true), T.sum(1 - y_true)]), dtype='int64')
    # indPos = indPos[:n_samples]
    # indNeg = indNeg[:n_samples]

    total = np.float64(patchSize * patchSize * patchZ)
    loss_vector = ifelse(T.gt(n_samples, 0),
                         # if this patch has positive samples, then calulate the first formula
                         (- alpha * T.sum(T.log(y_pred_clipped[indPos])) - (1 - alpha) * T.sum(
                             T.log(1 - y_pred_clipped[indNeg]))) / total,
                         - (1 - alpha) * T.sum(T.log(1 - y_pred_clipped[indNeg])) / total)

    average_loss = T.mean(loss_vector) / (1 - alpha)
    return average_loss


def validate(model, data):
    useSave = False
    if not useSave:
        Preds, Labels = validate_model(model=model, patchSize=patchSize, patchSize_out=patchSize_out, 
                                        patchZ=patchZ, patchZ_out=patchZ_out, modelType=modelType,
                                        bufferSize=bufferSize, save_predictions=filename+"_on_"+valid_name, dataset=data)
        np.save("/n/coxfs01/eric_wu/convlstm/np_data/pred_"+filename+".npy", (Preds, Labels))
    else:
        (Preds, Labels) = np.load("/n/coxfs01/eric_wu/convlstm/np_data/pred_"+filename+".npy")
    
    truth_seg = segment_gt_vesicle_style(prob = Labels,
                    sigma_xy = 0.0,
                    sigma_z = 0.0,
                    threshold = 0.0,
                    min_size_2d = 1,
                    max_size_2d = 1500000,
                    min_size_3d = 1,
                    min_slice = 1,
                    is_gt = True)

    scores = []
    min_2d = 150
    min_3d = 250
    max_2d = 100000
    print "min_2d", min_2d
    print "min_3d", min_3d

    #uniques = np.unique(Preds)
    #np.linspace(uniques[int(len(uniques)*0.4)], uniques[int(len(uniques)*0.6)], 6)
    saved = False
    for k in np.linspace(0.90, 0.99, 5):#np.linspace(0.99, 0.999, 5):
        print ""
        print "Evaluation for threshold", k
        print "======"
        #Preds = (nPreds).astype(np.uint64)

        prediction_seg = segment_pred_vesicle_style(prob = Preds,
                        sigma_xy = 0.0,
                        sigma_z = 0.0,
                        threshold = k,
                        min_size_2d = min_2d,
                        max_size_2d = max_2d,
                        min_size_3d = min_3d,
                        min_slice = 1,
                        is_gt = False)

        if not saved:
            print "Saving results"
            #Preds = np.zeros(Preds.shape)
            for i in range(10):#len(prediction_seg)):
                raw = normalize(Preds[i])
                pred = (prediction_seg[i]>0).astype(int)
                truth = normalize(Labels[i])
                #gray = normalize(data_validate[0][i])
                img_comp = np.concatenate((raw, pred, truth), axis=1)
                misc.imsave("./validation_results/"+filename+"_"+str("%04d"%i)+".png", img_comp)
            print "Results saved"
            saved = True

        num_predictions = len(np.unique(prediction_seg))-1
        num_truths = len(np.unique(truth_seg))-1
        print "Calculating Precision Recall"

        overlaps= sparse.csc_matrix((np.ones_like(truth_seg.ravel()), (truth_seg.ravel(), prediction_seg.ravel())))
        '''
        overlapb=overlaps[1:,1:]>0

        ngt,ndet=overlapb.shape
        colsum=overlapb.sum(axis=0)
        true_positive = np.sum(colsum==1)+np.sum(colsum>1)
        false_positive = ndet - true_positive
        false_negative = ngt - true_positive

        precision = true_positive*1./(true_positive+false_positive)
        recall = true_positive*1./(true_positive+false_negative)
        #scores += precision + "," + recall + "\n"
        scores.append([precision, recall])
        print "Precision:", precision
        print "Recall:", recall
        '''
        overlapsa = overlaps.todense()[1:,1:]
        overlaps_fixed = np.zeros(overlapsa.shape)
        for i in np.where(overlapsa.sum(axis=0)>0)[1]:
            overlaps_fixed[np.where(overlapsa[:,i]>0)[0][0],i] = 1.

        precision = np.mean((overlaps_fixed.sum(axis=0)>0).astype(int))
        print "Precision:", precision
        recall = np.mean((overlaps_fixed.sum(axis=1)>0).astype(int))
        print "Recall:", recall
        print "F1:", (2.*precision*recall)/(precision+recall)

        #segf1 = (2.*precision*recall)/(precision+recall)
        '''
        clefts_evaluation = Clefts(prediction_seg, truth_seg)
        (fp_count, adgt) = clefts_evaluation.false_positives(threshold=threshold)
        pos_count = np.where(prediction_seg>0)[0].shape[0]
        tp_count = pos_count-fp_count
        (fn_count, adf) = clefts_evaluation.false_negatives(threshold=threshold)

        cremi = np.mean([adgt,adf])
        scores.append((k, cremi))
        
        #print "false positives: " + str(fp_count)
        #print "false negatives: " + str(fn_count)
        precision = (tp_count*1.0/pos_count)
        recall = (tp_count*1.0/(tp_count+fn_count))
        print "precision: ", precision
        print "recall: ", recall
        pwf1 = (2.*precision*recall)/(precision+recall)
        
        #print "Seg F1 Score:", segf1
        #print "ADGT: " + str(adgt)
        #print "ADF: " + str(adf)
        #print "PW F1 Score:", pwf1
        #print "Cremi Score: ", cremi

        if doPrediction:
            for x in np.nditer(Preds, op_flags=['readwrite']):
                x[...] = 0xffffffffffffffff if x == 0 else 0x0000000000000000
            file = CremiFile("./np_data/"+filename+"_"+valid_name+".hdf", "w")
            clefts = Volume(Preds, resolution=(40.0, 4.0, 4.0), comment="sample 2d unet")
            file.write_clefts(clefts)
            file.close()
        '''
    np.savetxt("./results/scores_"+filename+"_"+valid_name+".txt", scores, delimiter=",")

print("Validating model:",filename)
print("For dataset:",valid_name)

print "Loading model", filename
model = model_from_json(open(models_dir + filename + '.json').read())

if weight_filename != "":
    model.load_weights(models_check_dir + weight_filename)
else:
    model.load_weights(models_dir + filename + '_weights.h5')

print "Compiling model", filename
opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#opt = SGD(lr=learn_rate, decay=0, momentum=momentum, nesterov=False)
model.compile(loss=weighted_mse, optimizer=opt)

print "Generating dataset"
data_validate = generate_dataset(dataset=valid_name, cropSize=cropSize, csZ=csZ, doDilation=doDilation)

validate(model, data_validate)
