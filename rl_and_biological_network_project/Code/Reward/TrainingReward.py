class TrainingReward:
    """Shaped reward for faster convergence during training"""
    def __init__(self):
        self.base_window = 10  # Start with wider time window
        self.scale = 0.5  # Reduced penalty/bonus scale
        
    def reward(self, response):
        total = 0
        for i in range(1, len(response)):
            dt = response[i,0] - response[i-1,0]
            
            # Progressive time window tightening
            window = max(5, self.base_window - int(i/1000))  # Example decay
            
            if dt > window:
                continue
                
            # Base reward for any paired spikes
            total += 0.2
            
            # Directional component (scaled)
            dir_diff = (response[i,1] - response[i-1,1]) % 4
            if dir_diff == 1:
                total += self.scale
            elif dir_diff == 3:
                total -= self.scale * 0.8  # Reduced penalty
                
        return total