class Parameters:
    def __init__(self):
        self.eps_upper = 10
        
        self.eps_lower = 0.01
        
        self.alpha = 0.5
        
        self.iter = 3

    def get_alpha(self):
        return self.alpha
    
    def get_eps(self):
        return [self.eps_upper, self.eps_lower]
    
    def get_iter(self):
        return self.iter