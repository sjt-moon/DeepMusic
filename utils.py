class TwoWayDictionary:
    def __init__(self):
        self.v2i = dict()
        self.i2v = dict()
    
    def __getitem__(self, key):
        return self.v2i[key]
    
    def __setitem__(self, key, value):
        self.v2i[key] = value
        self.i2v[value] = key
    
    def get(self, idx):
        return self.i2v[idx]
    
    def __str__(self):
        return str(self.v2i)
    
    def __repr__(self):
        return repr(self.v2i)
    
    def __len__(self):
        return len(self.v2i)
