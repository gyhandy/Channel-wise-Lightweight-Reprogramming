# Ghost_Res

## dataset
function to load dataset

## network
network structure

## run

```
python main.py [device #]:[task_id]:[model_name]:[batch_size]
```

### run h2t baseline

```
python head2toe.py [device #]:[task_id]:[model_name(not too important)]:[batch_size]
```

Note that to change hyperparameter, directly change line 16 to line 26. If you need to change any other parameter, optimizer, or scheduler, please contact me to makesure fair comparison is made among models. 