import math
import numpy as np
from collections.abc import Iterable

class PriorFactorScheduler():
    '''
    find the smallest prior factor that doesn't result in over-fitting
    by grid search or binary tree search
    it is assumed that a lower 'criterion' is better

    usage:
    1) init
    2) get_pf
    ... compute train and valid
    3) step
        step sends end signal

    use reset() to clear history
    '''

    def __init__(
        self, method, num, early_stopping, space, pf_range, test_no_regul,
        init_step_size=None, thresh=0.05, check_of_any=True):
        '''
        - method: 'grid' for grid search or 'tree' for binary tree search
        - num: number of prior factors to produce.
               stops proceeding and the 'done' flag becomes true when
               generates 'num' pfs.
        - space: log (base 10) or lin. search in logspace or lin space.
        - test_no_regul:first generate pf=0, then start generating from the range.
        - pf_range: range to search for pf in the form (min, max).
                    for grid, these are the min and max of interval.
                    both methods start at min, possibly after generating 0 (test_no_regul).
                    in both methods, pf always projected inside this interval.
                    NOTE: for log search space, min and max are not in log, i.e.,
                          the method searchs between (log(min), log(max)).
        - init_step_size: initial step size of the search algorithm.
                          for grid, the step size is always the same, unless pf hits max interval.
                          for tree, step size reduces if direction of search changes.
                          NOTE: log not applied on top of it if in log search space.
        - early_stopping: only used for 'grid'.
                          if no over-fitting, stop increasing pf and 'done'.
        - thresh: threshold for identifying over-fitting. no over-fitting if:
                  abs(valid_criterion-train_criterion) < thresh*train_criterion
        - check_of_any: avoid over-fitting for each client or on average
        '''
        assert method in ['grid', 'tree']
        assert space in ['lin', 'linear', 'log']
        if init_step_size is None:
            assert pf_range[0] <= pf_range[1]
        else:
            assert init_step_size+pf_range[0] <= pf_range[1]
        assert pf_range[0] <= pf_range[1]
        if space == 'log':
            assert pf_range[0] > 0
        else:
            pf_range[0] >= 0
        if (space in ['lin', 'linear']) and (pf_range[0]==0) and test_no_regul:
            test_no_regul = False
        # set attr
        self.method, self.num, self.thresh = method, num, thresh
        self.pf_range, self.early_stopping = pf_range, early_stopping
        self.test_no_regul, self.check_of_any = test_no_regul, check_of_any
        self.space = space if not space=='linear' else 'lin'
        if self.space == 'log':
            self.pf_range = (math.log2(self.pf_range[0]), math.log2(self.pf_range[1]))
        if init_step_size is None:
            if self.method == 'grid':
                if self.test_no_regul:
                    self.init_step_size = (1/(self.num-2))*(self.pf_range[1]-self.pf_range[0])
                else:
                    self.init_step_size = (1/(self.num-1))*(self.pf_range[1]-self.pf_range[0])
            elif self.method == 'tree':
                self.init_step_size = 0.5*(self.pf_range[1]-self.pf_range[0])
        else:
            self.init_step_size = init_step_size

        self.reset()


    def reset(self):
        self.msg = '[INFO] pf scheduler '
        # start at the min, increase pf
        self.dir = 1
        self.cur_num = 0
        self.done = False
        self.pfs = np.zeros(self.num)
        self.step_size = self.init_step_size
        if self.test_no_regul:
            self.pfs[0] = 0                 # start from 0
        else:
            self.pfs[0] = self.pf_range[0]  # start from the minimum


    def get_pf(self):
        assert not self.done
        if self.space =='log':
            if (self.cur_num==0) and self.test_no_regul:
                pf = self.pfs[self.cur_num]
            else:
                pf = 2**(self.pfs[self.cur_num])
        else:
            pf = self.pfs[self.cur_num]
        return pf


    def step(self, train_criterion, valid_criterion):
        assert not self.done
        train_criterion = np.array(train_criterion)
        valid_criterion = np.array(valid_criterion)

        # jump to min in range if started from 0
        if self.test_no_regul and self.cur_num==0:
            # check over-fit
            if self.early_stopping:
                if not self._check_over_fit(train_criterion=train_criterion, valid_criterion=valid_criterion):
                    self.done = True
                    self.msg += 'terminated because non-regularized model does not over-fit.'
                    return
            self.cur_num += 1
            self.pfs[self.cur_num] = self.pf_range[0]
            return

        # check if reached max num
        if self.cur_num==self.num-1:
            self.done=True
            self.msg += 'finished generating {:2.0f} prior factors.'.format(self.num)
            return

        self.cur_num += 1

        # fixed step size (grid)
        if self.method=='grid':
            # check ES
            if self.early_stopping:
                if not self._check_over_fit(train_criterion=train_criterion, valid_criterion=valid_criterion):
                    self.done = True
                    self.msg += 'found prior factor by grid search to avoid over-fitting.'
                    return
            self.dir = 1 # increase prior factor, no change in step size
            # compute next pf
            self.pfs[self.cur_num] = self.pfs[self.cur_num-1] + self.dir * self.step_size
            self.pfs[self.cur_num] = max(self.pfs[self.cur_num], self.pf_range[0])
            self.pfs[self.cur_num] = min(self.pfs[self.cur_num], self.pf_range[1])
            return


        # tree
        if self.method == 'tree':
            # unacceptably large validation criterion for at least one client or on average
            if self._check_over_fit(train_criterion=train_criterion, valid_criterion=valid_criterion):
                # check if increasing is possible
                if np.abs(self.pfs[self.cur_num-1]-self.pf_range[1])<1e-3:
                    self.done = True
                    self.msg += 'requires increasing the prior factor beyond the range - terminating.'
                    return
                self.dir = 1 # increase prior factor, no change in step size
            # valid and train are close => decrease regularization
            else:
                # check if decreasing is possible
                if np.abs(self.pfs[self.cur_num-1]-self.pf_range[0])<1e-3:
                    self.done = True
                    self.msg += 'requires decreasing the prior factor below the range - terminating.'
                    return
                self.dir = -1

            # divide step size by 2 until get a new pf
            candid_pf = min(max(self.pfs[self.cur_num-1]+self.dir*self.step_size, self.pf_range[0]), self.pf_range[1])
            while self._check_existing_pf(candid_pf):
                self.step_size = self.step_size/2
                candid_pf = min(max(self.pfs[self.cur_num-1]+self.dir*self.step_size, self.pf_range[0]), self.pf_range[1])

            self.pfs[self.cur_num] = candid_pf
            return



    def _check_over_fit(self, train_criterion, valid_criterion):
        if train_criterion is None or valid_criterion is None:
            return False
        if self.check_of_any:
            meas = (valid_criterion-train_criterion)/train_criterion # array of list num clients
            return (meas>self.thresh).any()
        else:
            train_criterion = train_criterion.reshape(-1)
            valid_criterion = valid_criterion.reshape(-1)
            return sum(valid_criterion-train_criterion)/sum(train_criterion) > self.thresh


    def _check_existing_pf(self, candid_pf):
        if not self.test_no_regul:
            if self.cur_num <= 0:
                return False
            else:
                return (np.abs(self.pfs[self.cur_num]-candid_pf)<self.step_size/2).any()
        else:
            if self.cur_num <= 1:
                return False
            else:
                return (np.abs(self.pfs[1:self.cur_num]-candid_pf)<self.step_size/2).any()






if __name__ == "__main__":
    train_criterions = [1, 1, 1, 1, 1, 1]
    valid_criterions = [2,2,2,2,2,2]

    pf_scheduler = PriorFactorScheduler(
                        method='tree', num=6, space='log', pf_range=(1e-2, 10),
                        test_no_regul=True)
    for tr, vl in zip(train_criterions, valid_criterions):
        pf = pf_scheduler.get_pf()
        print('\n pf = ', pf)
        pf_scheduler.step(tr, vl)
        print('step size ', pf_scheduler.step_size)
        print(pf_scheduler.msg)
        if pf_scheduler.done:
            break
