class EarlyStopper:
    '''
    Class to model early stopping monitor.
    Trigger a stop signal if a metric does not improve
    before patience is reached.
    '''
    
    MAX_CRITERION = 'max'
    MIN_CRITERION = 'min'
    STRICT_MAX_CRITERION = 'strict_max'
    STRICT_MIN_CRITERION = 'strict_min'
    
    def __init__(self, patience: int, criterion: str = 'min'):
        self.CRITERIONS = {
            self.MAX_CRITERION: self.__max_criterion_update, 
            self.MIN_CRITERION: self.__min_criterion_update,
            self.STRICT_MAX_CRITERION: self.__strict_max_criterion_update,
            self.STRICT_MIN_CRITERION: self.__strict_min_criterion_update
        }
        
        self.criterion = criterion
        if self.criterion == self.MAX_CRITERION or self.criterion == self.STRICT_MAX_CRITERION:
            self.best_value = -np.inf
        elif self.criterion == self.MIN_CRITERION or self.criterion == self.STRICT_MIN_CRITERION:
            self.best_value = np.inf
        
        self.patience = patience
        self.counter = 0
        
    def __max_criterion_update(self, metric) -> bool:
        return metric >= self.best_value
        
    def __min_criterion_update(self, metric) -> bool:
        return metric <= self.best_value
    
    def __strict_max_criterion_update(self, metric) -> bool:
        return metric > self.best_value
        
    def __strict_min_criterion_update(self, metric) -> bool:
        return metric < self.best_value
    
    def step(self, metric) -> bool:
        '''
        Update the best value.
        If the new metric is worse than the best store value,
        wrt to the chosen criterion, return True if the patience
        has been reached. Otherwise, return False.
        '''
        
        if self.CRITERIONS[self.criterion](metric):
            self.best_value = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter == self.patience - 1:
                return True
            
        return False