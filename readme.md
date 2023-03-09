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
## Not Support
```python
You may use nn.BatchNorm2d to replace torch.batch_norm
```

## 流程简述
add_binary_model
1. 对模型使用torch.fx获得计算节点，同时使用shapeprop将每个节点的输出获得
2. 对于每一个节点，通过名字得到对应节点的类型，目前是mul, add, pool三种默认通过节点名称获得，且将mul和add的节点加入到列表中
3. pool后面更的激活函数不增加dbc
4. 判断这个节点是不是激活函数节点
    1. 节点的类型分为三种，call_method, call_module, call_function，除了call_module外，基本都可以通过节点的属性判断是否为激活函数
    2. 若为call_module，再通过torch.fx获得module内部信息，来判断是否是激活函数，同时可以获得这个激活函数对应的名字
5. 通过上述步骤，得到了node_list_activate，是一个字典，key为激活函数的node，value为激活函数的名称
6. 判断要替换的节点在不在add的历史节点里，在add的历史节点内的激活函数是没办法增加dbc [add后的激活函数不增加binary_conv2d，因为无法找到对应的model_op]
7. 判读激活函数前的节点是否是group_conv，如果是则剔除(考虑conv-bn-act, conv-act两种)
8. node前的节点不能是mul或者silu,最多遍历两层
9. 获得了最终可以替换的激活函数节点
10. 通过激活函数节点，找到激活函数对应的conv
11. 通过fx_model.graph.inserting_after增加对应的dbc
12. 返回新的模型，以及需要替换的conv，在rebuild时使用