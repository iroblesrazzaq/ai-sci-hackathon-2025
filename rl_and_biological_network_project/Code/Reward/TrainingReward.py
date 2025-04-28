class TrainingReward:
    """Shaped reward for faster convergence during training"""
    def __init__(self):
        self.base_window = 10  # Start with wider time window
        self.iter = 0 # Used to progressively tighten the window
        
    def reward(self, response):
        total = 0
        for i in range(2, len(response)):
            dt = response[i,0] - response[i-1,0]
            
            # Progressive time window tightening
            window = max(5, self.base_window - int(self.iter/1000))
            
            if dt > window:
                continue
            
            # Directional component
            dir_diff = (response[i,1] - response[i-1,1]) % 4
            if dir_diff == 1:
                total += 1
            elif dir_diff == 3:
                total -= 1
                
        self.iter += 1
                
        return total