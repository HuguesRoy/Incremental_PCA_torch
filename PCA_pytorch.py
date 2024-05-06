#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
##################
#  adapted from : https://github.com/gngdb/pytorch-pca/blob/main/pca.py
#                 https://github.com/scikit-learn/scikit-learn/blob/f07e0138b/sklearn/decomposition/_incremental_pca.py#L19


def svd_flip(u, v, u_based_decision ):
    # columns of u, rows of v
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u.T), 1).to(u.device)
        i = torch.arange(u.T.shape[0]).to(u.device)
        signs = torch.sign(u[max_abs_cols, i])
        u *= signs
        v *= signs.view(-1, 1)
    else:
        max_abs_v_rows = torch.argmax(torch.abs(v), axis=1).to(u.device)
        shift = torch.arange(v.shape[0]).to(u.device)
        indices = max_abs_v_rows + shift * v.shape[1]
        signs = torch.sign(torch.take(torch.reshape(v, (-1,)), indices))
        u *= signs
        v *= signs.view(-1,1)

    return u, v

class PCA(nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components

    @torch.no_grad()
    def fit(self, X):
        n, d = X.size()
        if self.n_components is not None:
            d = min(self.n_components, d)
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        Z = X - self.mean_ # center
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        Vt = Vh
        U, Vt = svd_flip(U, Vt)
        self.register_buffer("components_", Vt[:d])
        return self
    
    def forward(self, X):
        return self.transform(X)

    def transform(self, X):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(X - self.mean_, self.components_.t())

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_


# Modified to be Incremental PCA
from torch.utils.data import DataLoader

class IncrementalPCA(nn.Module):

    def __init__(self, n_components : int, batch_size :int = None, device : str = 'cpu'):
        super().__init__()
        self.n_components = n_components
        self.batch_size = batch_size
        self.device = device

        
       

    @torch.no_grad()
    def fit(self, Dataset, num_workers :int = 0):

        self.register_buffer("components_", None)
        self.register_buffer("n_samples_seen_", torch.tensor(0))
        self.register_buffer("mean_", torch.tensor(0.0))
        self.register_buffer("var_", torch.tensor(0.0))
        self.register_buffer("singular_values_",None)
        self.register_buffer("explained_variance_",None)
        self.register_buffer("explained_variance_ratio_", None)
        self.register_buffer("noise_variance_",None)
        
        self.returnU = None

        

        total_batch = np.ceil(len(Dataset)//self.batch_size)
        #create a dataloader to handle batch size (for example if 10 samples and batch_size 3, we will have 3,3,3,1)
        #the last batch (1 in this example) will not be taken into account if batch_size < n_components
        PCA_loader = DataLoader(
            Dataset,
            batch_size= self.batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory= True
        )

        for i, batch in enumerate(PCA_loader):
            # for each batch compute partial fit that will update each attributes
            print(f'iteration : {i} / {total_batch}')

            self.partial_fit(batch.to(self.device))

        
        
        return self
    
    @torch.no_grad()
    def partial_fit(self, X, y=None):
        
        first_pass = not hasattr(self, "components_")

        n_samples, n_features = X.size()

        # if attribute not defined
        if first_pass:
            self.components_ = None
        
        # assign the number of components depending of the size of n_samples, n_features
        
        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not self.n_components <= n_features:
            raise ValueError(
                "n_components=%r invalid for n_features=%d, need "
                "more rows than columns for IncrementalPCA "
                "processing" % (self.n_components, n_features)
            )
        elif not self.n_components <= n_samples:
            pass #pass because usually the dataloader will end with smaller btach size than number of components
            # raise ValueError(
            #     "n_components=%r must be less or equal to "
            #     "the batch number of samples "
            #     "%d." % (self.n_components, n_samples)
            # )
        else:
            self.n_components_ = self.n_components


        # This is the first partial_fit
        if not hasattr(self, "n_samples_seen_"):
            self.n_samples_seen_ = 0
            self.mean_ = 0.0
            self.var_ = 0.0


        last_sample_count = torch.repeat_interleave(self.n_samples_seen_,  X.size()[1])
        
        # Update stats - they are 0 if this is the first step
        col_mean, col_var, n_total_samples = _incremental_mean_and_var(
            X,
            last_mean=self.mean_,
            last_variance=self.var_,
            last_sample_count= last_sample_count.to(self.device),
        )

        n_total_samples = n_total_samples[0]
        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = torch.sqrt(
                (self.n_samples_seen_ / n_total_samples) * n_samples
            ) * (self.mean_ - col_batch_mean)
            X = torch.vstack(
                (
                    self.singular_values_.reshape((-1, 1)) * self.components_,
                    X,
                    mean_correction,
                )
            )

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples)

        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.returnU = U[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        # we already checked `self.n_components <= n_samples` above
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_ :].mean()
        else:
            self.noise_variance_ = torch.tensor(0.0)
        return self
    
    def forward(self, X):
        return self.transform(X)

    def transform(self, Dataset, batch_size : int = None ):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        print(self.device)

        if batch_size is None:
            batch_size = self.batch_size

        final_tensor = torch.empty((0,self.n_components)).to(self.device)

        PCA_loader = DataLoader(
            Dataset,
            batch_size= batch_size,
            shuffle = False,
            num_workers = 0,
            pin_memory= True
        )
        total_batch = np.ceil(len(Dataset)//self.batch_size)

        for i,batch in enumerate(PCA_loader):
            print(f'iteration : {i} / {total_batch}')
            temp = torch.matmul(batch.to(self.device) - self.mean_.to(self.device), self.components_.t().to(self.device))
            final_tensor = torch.cat((final_tensor, temp))

        return final_tensor

    def fit_transform(self, Dataset):
        self.fit(Dataset)
        return self.transform(Dataset)

    def inverse_transform(self, Y):
        assert hasattr(self, "components_"), "PCA must be fit before use."
        return torch.matmul(Y, self.components_) + self.mean_
    
    def load_dict(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)
            



def _incremental_mean_and_var(
    X, last_mean, last_variance, last_sample_count, sample_weight=None
):
    """Calculate mean update and a Youngs and Cramer variance update.

    If sample_weight is given, the weighted mean and variance is computed.

    Update a given mean and (possibly) variance according to new data given
    in X. last_mean is always required to compute the new mean.
    If last_variance is None, no variance is computed and None return for
    updated_variance.

    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data to use for variance update.

    last_mean : array-like of shape (n_features,)

    last_variance : array-like of shape (n_features,)

    last_sample_count : array-like of shape (n_features,)
        The number of samples encountered until now if sample_weight is None.
        If sample_weight is not None, this is the sum of sample_weight
        encountered.

    sample_weight : array-like of shape (n_samples,) or None
        Sample weights. If None, compute the unweighted mean/variance.

    Returns
    -------
    updated_mean : ndarray of shape (n_features,)

    updated_variance : ndarray of shape (n_features,)
        None if last_variance was None.

    updated_sample_count : ndarray of shape (n_features,)

    Notes
    -----
    NaNs are ignored during the algorithm.

    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247

    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    X_nan_mask = torch.isnan(X)
    if torch.any(X_nan_mask):
        sum_op = torch.nansum
    else:
        sum_op = torch.sum


    new_sum = _safe_accumulator_op(sum_op, X, axis=0)
    n_samples = X.size()[0]
    new_sample_count = (n_samples - torch.sum(X_nan_mask, axis=0))


    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        T = new_sum / new_sample_count
        temp = X - T

        correction = _safe_accumulator_op(sum_op, temp, axis=0)
        temp **= 2
        new_unnormalized_variance = _safe_accumulator_op(sum_op, temp, axis=0)

        # correction term of the corrected 2 pass algorithm.
        # See "Algorithms for computing the sample variance: analysis
        # and recommendations", by Chan, Golub, and LeVeque.
        new_unnormalized_variance -= correction**2 / new_sample_count

        last_unnormalized_variance = last_variance * last_sample_count

        with np.errstate(divide="ignore", invalid="ignore"):
            last_over_new_count = last_sample_count / new_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum) ** 2
            )

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count

# Adapted for torch

# Use at least float64 for the accumulating functions to avoid precision issue
# see https://github.com/numpy/numpy/issues/9393. The float64 is also retained
# as it is in case the float overflows
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.

    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    if torch.is_floating_point(x) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype= torch.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


if __name__ == "__main__":
    

    from torch.utils.data import Dataset

    class IrisDataset(Dataset):
        def __init__(self, array,transform =None):
            self.transform = transform
            self.array = torch.from_numpy(array)

        def __len__(self):
            return self.array.shape[0]

        def __getitem__(self, idx):
            return self.array[idx,:]

    import numpy as np
    from sklearn.decomposition import IncrementalPCA as sklearn_PCA
    from sklearn import datasets
    iris = torch.tensor(datasets.load_iris().data)
    _iris = iris.numpy()
    torch_iris = IrisDataset(_iris)
    devices = ['cpu']
    batch_size = 20
    if torch.cuda.is_available():
        devices.append('cuda')
    for device in devices:
        iris = torch_iris
        for n_components in (2, 4):
            l_pca = sklearn_PCA(n_components=n_components, batch_size = batch_size).fit(_iris)
            l_components = torch.tensor(l_pca.components_)
            pca_inc = IncrementalPCA(n_components=n_components, batch_size= batch_size, device =device).to(device).fit(torch_iris)
            components = pca_inc.components_

            assert torch.allclose(components, l_components.to(device))
            l_t = torch.tensor(l_pca.transform(_iris))
            t = pca_inc.transform(iris)
            assert torch.allclose(t, l_t.to(device))
        __iris = pca_inc.inverse_transform(t)
        iris = torch.from_numpy(l_pca.inverse_transform(l_t)).to(device)
        assert torch.allclose(__iris, iris)


        param_dict = pca_inc.state_dict()
        pca = IncrementalPCA(n_components=n_components, batch_size= batch_size, device =device)
        pca.load_dict(param_dict)

    print("passed!")






