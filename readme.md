## How to use
```python

from pas_method import pas
from pas_method.rebuild import rebuild_model

pas_init = pas.PaS(model=net, input_shape=(1, 3, 32, 32), prune_ratio=0.5)
net, bn_names = pas_init.init()

#traning
...


# after training end
net_ori = crate_model(...) #your create model function
pas_init = pas.PaS(model=net_ori, input_shape=(2, 3, 224, 224), prune_ratio=0.5)
net, bn_names = pas_init.init()
net = net.to("cuda")

weights = torch.load(your_weights_path)
net.load_state_dict(weights)

# rebuild model which remove dbc and remove channel
model_rebuild = rebuild_model(ori_model=net_ori, dbc_model=net, dbc_weights=weights, bn_names=bn_names,
                              input_shape=[1, 3, 224, 224])

```
