import torch, copy, math
import numpy as np

class SyNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, random_seed):
        torch.manual_seed(random_seed)
        super(SyNet, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim, dtype=torch.double)

    def forward(self, x):
        x = self.linear(x)
        return x

    def predict(self, X):
        return self(X)

def print_param(model, message=''):
    with np.printoptions(precision=3, suppress=True):
        print(message+' model: \nbias = ' +
              str(model.state_dict()['linear.bias'].numpy()[0]) +
              ', \nweights = ' +
              str(model.state_dict()['linear.weight'].numpy().flatten()))
    return


def get_av_weights(households, **kwargs):
    models = kwargs.get('models', [household.model_mtl for household in households])
    in_dim = households[0].info['num_features']
    # find total samples
    tot_samp = 0
    for household in households:
        tot_samp = tot_samp + household.info['train_samples']
    # init
    bias = np.zeros(1)
    wght = np.zeros(in_dim)
    # aggregate
    for model in models:
        # aggregate updates
        bias_hh = model.state_dict()['linear.bias'].numpy()
        wght_hh = model.state_dict()['linear.weight'].numpy().flatten()
        bias = bias_hh*household.info['train_samples']/tot_samp + bias
        wght = wght_hh*household.info['train_samples']/tot_samp + wght
    return bias, wght


def mtl_train(households, lr, lambda_, inner_iters, outer_iters, optim_method, random_seed, **kwargs):
    '''
    optim_method: Adam or SGD
    '''
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    verbose = kwargs.get('verbose', False)
    epsilon = kwargs.get('epsilon', 100)
    delta = kwargs.get('delta', 1)
    gamma = kwargs.get('gamma', 1)
    priv = kwargs.get('priv', False)
    use_gamma = kwargs.get('use_gamma', True)
    # noise
    if priv:
        sigma = 4 * gamma * math.sqrt(outer_iters * math.log(1/delta)) / (epsilon * len(households))
        print('private, using noise std of ' + str(sigma))

    # find total samples
    tot_samp = 0
    for household in households:
        tot_samp = tot_samp + household.info['train_samples']

    # initialize mtl for households
    num_households = len(households)
    init_state_dicts=[]
    for household in households:
        household.mtl_init(lr)
        init_state_dicts.append(copy.deepcopy(household.model_mtl.state_dict()))
    # initialize w_0
    in_dim=households[0].info['num_features']
    w_0 = SyNet(#torch,
        in_dim=in_dim, out_dim=1, random_seed=random_seed)
    # find av initial weights
    init_bias, init_wght = get_av_weights(households)
    w_0.state_dict()['linear.weight'].copy_(torch.tensor(init_wght.reshape((1,len(init_wght)))))
    w_0.state_dict()['linear.bias'].copy_(torch.tensor(init_bias))

    # create optimizer
    if optim_method=='Adam':
        optim = torch.optim.Adam(params=w_0.parameters(), lr=lr)
    else:
        if optim_method=='SGD':
            optim = torch.optim.SGD(params=w_0.parameters(), momentum=0, lr=lr)
        else:
            print('Unsupported optimization method')
            return

    # updates vec
    g = np.zeros((num_households, outer_iters, in_dim))
    g_clipped = np.zeros((num_households, outer_iters, in_dim))
    g_noisy = np.zeros((num_households, outer_iters, in_dim))
    g_norm = np.zeros((num_households, outer_iters))
    g_clipped_norm = np.zeros((num_households, outer_iters))
    g_noisy_norm = np.zeros((num_households, outer_iters))

    # iterate
    for i in np.arange(outer_iters):
        if verbose:
            print('\n before iter ' + str(i))
            print_param(w_0, 'w_0 ')

        # sample noise
        n = np.zeros(in_dim)
        if priv:
            n = np.random.normal(0, sigma, in_dim)

        # initialize param update
        cur_state_dict=copy.deepcopy(w_0.state_dict())
        w_0_wght = w_0.state_dict()['linear.weight']
        w_0_bias = w_0.state_dict()['linear.bias']
        delta_bias = np.zeros(1)
        delta_wght = np.zeros(in_dim)
        # run minibatch SGD for each household
        for h_num, household in enumerate(households):
            # run minibatch SGD and get update in parameters
            db, dw = household.mtl_iterate(w_0_wght=w_0_wght, w_0_bias=w_0_bias,
                                           inner_iters=inner_iters,
                                           lambda_=lambda_, verbose=False)
            # record update
            g[h_num][i][:] = dw
            g_norm[h_num][i] = np.linalg.norm(g[h_num][i][:])
            # clip updates
            if use_gamma:
                g_clipped[h_num][i][:] = g[h_num][i][:]* min(1,gamma/g_norm[h_num][i])
            else:
                g_clipped[h_num][i][:] = g[h_num][i][:]
            g_clipped_norm[h_num][i] = np.linalg.norm(g_clipped[h_num][i][:])
            # noisy updates
            #g_noisy_norm[h_num][i] = np.linalg.norm(g_clipped[h_num][i][:]+n/num_households)
            g_noisy[h_num][i][:] = g_clipped[h_num][i][:] + n/num_households
            g_noisy_norm[h_num][i] = np.linalg.norm(g_noisy[h_num][i][:])
            if not (g_noisy[h_num][i][:]-n/num_households==g_clipped[h_num][i][:]).any:
                print('[err]')

            # aggregate updates
            delta_bias = db*household.info['train_samples']/tot_samp + delta_bias
            delta_wght = g_clipped[h_num][i][:]*household.info['train_samples']/tot_samp + delta_wght # noise is added at the end
            # reset w_0
            for key, value in cur_state_dict.items():
                w_0.state_dict()[key].copy_(value)

        # update w_0
        new_bias = w_0_bias + torch.tensor(delta_bias.reshape((1)))
        new_wght = w_0_wght + torch.tensor(delta_wght.reshape((1,in_dim)))
        new_wght = new_wght + torch.tensor(n.reshape((1,in_dim)))
        # set
        w_0.state_dict()['linear.weight'].copy_(new_wght)
        w_0.state_dict()['linear.bias'].copy_(new_bias)

        if use_gamma:
            if np.linalg.norm(delta_wght > gamma):
                print('error')

        result=dict()
        result['w_0']=w_0
        result['init_state_dicts']=init_state_dicts
        result['g']=g
        result['g_clipped']=g_clipped
        result['g_noisy']=g_noisy
        result['g_norm']=g_norm
        result['g_clipped_norm']=g_clipped_norm
        result['g_noisy_norm']=g_noisy_norm
    return result



def mtl_employ(households, trained_lambda, trained_w_0, num_iters, lr, random_seed):
    '''
    employ trained MTL mean model
    optim_method: Adam or SGD
    '''
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # initialize mtl for households
    num_households = len(households)
    for household in households:
        household.mtl_init(lr)

    w_0_wght = trained_w_0.state_dict()['linear.weight']
    w_0_bias = trained_w_0.state_dict()['linear.bias']


    # run minibatch SGD for each household
    for h_num, household in enumerate(households):
        # run minibatch SGD and get update in parameters
        _, _ = household.mtl_iterate(
            w_0_wght=w_0_wght, w_0_bias=w_0_bias,
            inner_iters=num_iters, lambda_=trained_lambda, verbose=False)

    # houshold.model_mtl is trained