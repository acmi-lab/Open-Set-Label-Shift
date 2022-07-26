from dis import dis
import numpy as np 
import torch
import torch.nn.functional as F

def get_label_dist(labels, num_classes):
    """
    Compute the label distribution of the data.
    """
    # Compute the label distribution of the data
    # Inputs:
    # - labels: Numpy array of shape (N,) giving the ground truth labels of
    #   the target data.
    # - num_classes: Number of classes in the source dataset.
    #
    # Outputs:
    # - label_dist: Numpy array of shape (num_classes ,) giving the label distribution
    #   of the data.

    label_dist = np.zeros(num_classes)

    for i in range(num_classes):
        label_dist[i] = np.mean(labels == i)

    return label_dist


def DKW_bound(x,y,t,m,n,delta=0.1, gamma= 0.01):
    temp = np.sqrt(np.log(1/delta)/2/n) + np.sqrt(np.log(1/delta)/2/m)
    bound = temp*(1+gamma)/(y/n)

    estimate = t

    return estimate, t - bound, t + bound

def top_bin_estimator(pdata_probs, udata_probs):
    """
    Compute the top bin estimator count.
    """
    # Compute the top bin estimator count
    # Inputs:
    # - pdata_probs: Numpy array of shape (N,) giving the probabilities
    #   of the source data.
    # - udata_probs: Numpy array of shape (N, ) giving the probabilities
    #   of the target data.
    #
    # Outputs:
    # - top_bin_estimator: scalar with estimate of pdata_probs samples in udata_probs. 
    # - upper_bound: scalar with upper bound of the estimate. 
    # - lower_bound: scalar with lower bpund of the estimate.

    p_indices = np.argsort(pdata_probs)
    sorted_p_probs = pdata_probs[p_indices]
    sorted_p_probs = sorted_p_probs[::-1]

    u_indices = np.argsort(udata_probs)
    sorted_u_probs = udata_probs[u_indices]
    sorted_u_probs = sorted_u_probs[::-1]
    
    num = len(sorted_u_probs)
    i = 0
    j = 0 
    upper_cfb = []
    lower_cfb = []
    estimate_arr = []

    while (i < num):
        start_interval =  sorted_u_probs[i]   
        if (i<num-1 and start_interval> sorted_u_probs[i+1]): 
            pass
        else: 
            i += 1
            continue

        while ( j<len(sorted_p_probs) and sorted_p_probs[j] >= start_interval):
            j+= 1

        if j>1 and i > 1:
            t = (i)*1.0*len(sorted_p_probs)/j/len(sorted_u_probs)
            estimate, lower , upper = DKW_bound(i, j, t, len(sorted_u_probs), len(sorted_p_probs))
            estimate_arr.append(estimate)
            upper_cfb.append( upper)
            lower_cfb.append( lower)

        i+=1

    if (len(upper_cfb) != 0): 
        idx = np.argmin(upper_cfb)
        return estimate_arr[idx], upper_cfb[idx], lower_cfb[idx]

    else:
        return 0, 0, 0


def BBE_estimate_multiclass(source_probs, source_labels, target_probs, num_classes):
    """
    Compute BBE estimate of the classifier.
    """
    # Compute the BBE estimate of the classifier
    # Inputs:
    # - source_probs: Numpy array of shape (N, 2*num_classes) giving the probabilities
    #   of the source data.
    # - source_labels: Numpy array of shape (N,) giving the ground truth labels of
    #   the source classifier.
    # - target_probs: Numpy array of shape (N, 2*num_classes) giving the probabilities
    #   of the target data.
    # - num_classes: Number of classes in the source dataset.
    #
    # Outputs:
    # - MP_estimate: Numpy array of shape (num_classes + 1,) giving the MP estimate of
    #   for each class.

    MP_estimate = []
    estimate_sum = 0.0 

    for i in range(num_classes):
        source_idx_i = np.where(source_labels == i)[0]
        source_probs_i = source_probs[source_idx_i, i]

        target_probs_i = target_probs[:, i]

        estimate_i, _, _ = top_bin_estimator(source_probs_i, target_probs_i)
        MP_estimate.append(estimate_i)
        estimate_sum += estimate_i


    MP_estimate.append(max(1.0 -estimate_sum, 0.0))

    return np.array(MP_estimate)

def BBE_estimate_binary(source_probs, target_probs):
    """
    Compute BBE estimate of the classifier.
    """
    # Compute the BBE estimate of the classifier
    # Inputs:
    # - source_probs: Numpy array of shape (N, ) giving the probabilities
    #   of the source data.
    # - target_probs: Numpy array of shape (N,) giving the probabilities
    #   of the target data.
    #
    # Outputs:
    # - estimate: scalar with the MP estimate of the class.

    estimate, _, _ = top_bin_estimator(source_probs, target_probs)

    return estimate


def estimator_CM_EN(pdata_probs, pudata_probs):
    return np.sum(pudata_probs)*len(pdata_probs)/len(pudata_probs)/np.sum(pdata_probs)

def estimator_prob_EN(pdata_probs):
    return np.sum(pdata_probs, axis=0)/len(pdata_probs)

def estimator_max_EN(pdata_probs, pudata_probs):
    return np.max(np.concatenate((pdata_probs, pudata_probs) ))

def keep_samples(probs, idx,  MP_estimate, num_classes):
    """
    Compute the keep samples for the target data.
    """
    # Compute the keep samples for the classifier
    # Inputs:
    # - probs: Numpy array of shape (N, 2*num_classes) giving the probabilities
    #   of the target data.
    # - idx: Numpy array of shape (N,) giving the indices of the data.
    # - MP_estimate: Numpy array of shape (num_classes + 1,) giving the MP estimate of
    #   for each class.
    # - num_classes: Number of classes in the source dataset.
    #
    # Outputs:
    # - keep_samples: Numpy array of shape (N, num_classes) giving the keep samples for each
    #   target sample.



    keep_samples = np.zeros((probs.shape[0], num_classes))
    num_samples = probs.shape[0]

    for i in range(num_classes):
        alpha_i = MP_estimate[i]
        sorted_i_idx = np.argsort(probs[:, i])
        sorted_original_idx = idx[sorted_i_idx]

        keep_original_idx = sorted_original_idx[: num_samples -  int(alpha_i*num_samples)]
        keep_samples[keep_original_idx, i] = 1

    return keep_samples


def keep_samples_discriminator(probs, idx,  MP_estimate):
    """
    Compute the keep samples for the target data.
    """
    # Compute the keep samples for the classifier
    # Inputs:
    # - probs: Numpy array of shape (N) giving the probabilities
    #   of the target data.
    # - idx: Numpy array of shape (N,) giving the indices of the data.
    # - MP_estimate: Scalar giving the MP estimate of the OOD class
    #
    # Outputs:
    # - keep_samples: Numpy array of shape (N) giving the keep samples for each
    #   target sample.

    num_samples = probs.shape[0]

    sorted_idx = np.argsort(probs)
    sorted_original_idx = idx[sorted_idx]

    keep_original_idx = sorted_original_idx[-int(MP_estimate*num_samples):]
    keep_samples = np.zeros(probs.shape[0])
    keep_samples[keep_original_idx] = 1

    return keep_samples

def transform_probs_EN(target_probs,disc_target_probs, y_s, y_t, num_classes):
    """
    Transform the target probabilities to the expected number of samples.
    """
    # Transform the target probabilities to the expected number of samples.
    # Inputs:
    # - target_probs: Numpy array of shape (N, num_classes) giving the probabilities
    #   of the target data.
    # - disc_target_probs: Numpy array of shape (N, ) giving the probabilities
    #   of the target data.
    # - y_s: Numpy array of shape (N,) giving the ground truth labels of
    #   the source data.
    # - y_t: Numpy array of shape (N,) giving the ground truth labels of
    #   the target data.
    # - num_classes: Number of classes in the source dataset.
    #
    # Outputs:
    # - transformed_target_probs: Numpy array of shape (N, num_classes + 1) giving the transformed
    #   target probabilities.

    transformed_target_probs = np.zeros((target_probs.shape[0], num_classes + 1))

    prob_s_i = np.zeros(num_classes)

    for i in range(num_classes):
        count_s_i = np.sum(y_s == i)
        count_t_i = np.sum(y_t == i)
        prob_s_i[i] = count_s_i*1.0/(count_s_i + count_t_i)

    for i in range(num_classes):
        transformed_target_probs[:, i ] = (1- prob_s_i[i]) * (disc_target_probs) * target_probs[:, i] /prob_s_i[i] / (1- disc_target_probs)


    transformed_target_probs[:, num_classes] = 1.0 - np.sum(transformed_target_probs[:, 0:num_classes], axis=1)

    return transformed_target_probs

def resample_probs(probs, labels, label_dist):
    """
    Resample the probabilities.
    """
    # Resample the probabilities.
    # Inputs:
    # - probs: Numpy array of shape (N, ) giving the probabilities
    #   of the target data.
    # - labels: Numpy array of shape (N,) giving the ground truth labels of
    #   the target data.
    # - label_dist: Numpy array of shape (num_classes,) giving the distribution
    #   of the target labels.
    #
    # Outputs:
    # - resampled_probs: Numpy array of shape (*, num_classes) giving the resampled
    #   target probabilities.

    num_samples = probs.shape[0]
    num_classes = label_dist.shape[0]

    # Assuming uniform source prior
    # max_prob = np.max(label_dist)*num_classes

    resample_idx = []


    for i in range(num_classes):
        idx_i = np.where(labels == i)[0]
        keep_idx_i = np.random.choice(idx_i, int(label_dist[i]*num_samples), replace=True)
        resample_idx.append(keep_idx_i)

    resample_idx = np.concatenate(resample_idx)

    return resample_idx 

def inverse_softmax(preds):
	preds[preds==0.0] = 1e-40
	preds = preds/np.sum(preds, axis=1)[:, None]
	return np.log(preds) - np.mean(np.log(preds),axis=1)[:,None]

def idx2onehot(a, k): 
	a = a.astype(int)
	b = np.zeros((a.size, k))
	b[np.arange(a.size),a] = 1
	
	return b

def label_shift_correction(probs, label_dist): 

    temp_probs = probs*label_dist[None]

    temp_probs = temp_probs/ np.sum(temp_probs, axis=1)[:, None]

    return temp_probs

def entropy(p):
    p = F.softmax(p, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p+1e-5), 1))

def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p, dim=1)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-5), 1)-value), margin))


def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)

def sigmoid_loss(out, y): 
    # loss = torch.gather(out, dim=1, index=y).sum()
    loss = out.gather(1, 1- y.unsqueeze(1)).mean()
    return loss
    
def log_everything(log_file, epoch, target_label_shift_acc="NA", target_orig_acc="NA", \
    target_seen_label_acc="NA", target_seen_acc="NA", source_acc="NA", \
    precision="NA", recall="NA", domain_disc_acc = "NA", domain_disc_valid_acc="NA", \
    target_marginal_estimate="NA", target_marginal ="NA"):

    if target_marginal_estimate == "NA":
        estimation_error = "NA"
        ood_estimation_error = "NA"
    else:
        estimation_error = np.sum(np.abs(target_marginal_estimate[:-1] - target_marginal[:-1]))
        ood_estimation_error = np.sum(np.abs(target_marginal_estimate[-1] - target_marginal[-1]))

    with open(log_file, "a") as f:
        f.write(f"{epoch},{target_label_shift_acc},{target_orig_acc}, {target_seen_label_acc},{target_seen_acc},{source_acc}, {precision},{recall},{domain_disc_acc},{domain_disc_valid_acc},{estimation_error},{ood_estimation_error}\n")

