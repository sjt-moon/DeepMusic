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


def get_hidden_states(musicfile, net, trainer):
    with open(musicfile) as infile:
        content = infile.read()
    string = '`' + string + '$'
    net.hidden = trainer.prepare_hidden(1)
    
    hidden_states = []
    for ch in string:
        seqs = [map(ord, ch)]
        inputs, _ = trainer.prepare_inputs_targets(seqs)
        _ = net(inputs, [1], per_char_generation=True)
        net.update_hidden()
        hidden_states.append(net.hidden.data.cpu())
    return hidden_states
