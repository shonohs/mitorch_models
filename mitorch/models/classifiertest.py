import collections
import functools
import torch

class Classifier(torch.nn.Module):
    def __init__(self, num_classes, feature_planes):
        assert isinstance(num_classes, int) or isinstance(num_classes, collections.abc.Iterable)
        super(Classifier, self).__init__()
        self.feature_planes = feature_planes
        if isinstance(num_classes, int):
            num_classes = (num_classes,)
        self.num_classes = num_classes

        self.features = None # Need to be initialized by a child class.
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(feature_planes, c) if c > 0 else torch.nn.Identity() for c in num_classes])

        self.outputs = []

    def forward(self, x, classifier_index=0):
        if isinstance(x, list):
            sizes = [y.size()[0] for y in x]
            x = torch.cat(x, 0)
            out = self.features(x)
            out = out.view(-1, self.feature_planes)
            outs = torch.split(out, sizes, dim=0)
            return [self.fcs[i](out) for i, out in enumerate(outs)]
        else:
            assert classifier_index < len(self.fcs)
            out = self.features(x)
            out = out.view(-1, self.feature_planes)
            return self.fcs[classifier_index](out)

    def reset_parameters(self):
        for m in self.fcs.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
        for m in self.features.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _forward_hook(self, module, input, output, index):
        self.outputs[index] = output
    
    def get_outputs(self, x, layer_names):
        handles = []
        for i, layer_name in enumerate(layer_names):
            layer = self.find_layer_by_name(layer_name)
            handles.append(layer.register_forward_hook(functools.partial(self._forward_hook, index=i)))
            
        self.outputs = [None] * len(layer_names)

        # Some subclass overwrites forward() method, which can call get_outputs() method.
        Classifier.forward(self, x)

        # Remove the hooks
        for handle in handles:
            handle.remove()

        assert all([o is not None for o in self.outputs])
        
        return self.outputs

    def find_layer_by_name(self, name):
        paths = name.split('.')
        current = self
        for p in paths:
            children = current.named_children()
            for name, c in children:
                if name == p:
                    current = c
                    break
        return current

    def make_feature_extractor(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def load_feature_extractor_state(self, state_dict):
        filtered_dict = {}
        features_state_dict = self.features.state_dict()
        for name in features_state_dict.keys():
            name = 'features.' + name
            if name in state_dict:
                filtered_dict[name] = state_dict[name]
            else:
                print("Warning: {} is missing".format(name))
                
        self.load_state_dict(filtered_dict, strict=False)
        
        
class JustClassifier(Classifier):
    def __init__(self, num_classes, feature_planes):
        super(JustClassifier, self).__init__(num_classes, feature_planes)
        self.features = lambda x: x
