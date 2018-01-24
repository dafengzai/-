这个项目是机器学习工程师纳米学位的最终项目，该项目源自于Kaggle 比赛Rossmann Store Sales 中,Rossmann 是欧洲一家连锁药店，我们需要根据过去各个Rossmann 药妆店每日的销售情况及相关信息情况，来预测Rossmann 未来3个月的销售额。  

本人使用xgboost模型对数据进行预测，本项目使用的库总共有：

```
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
```

进行feature engineering后在我的笔记本上(5-4210M)训练最终模型的时间约为4个小时，运行代码为：

```
num_boost_round = 4000

print("Train a XGBoost model")
dtrain = xgb.DMatrix(x_train[features], y_train)
dvalid = xgb.DMatrix(x_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=500, feval=rmspe_xg, verbose_eval=True)
```

![snap_screen_20171227200441](https://github.com/dafengzai/Udacity-Machine-Learning-Engineer-Nanodegree-Project/tree/master/capstone_project/MarkdownImages\snap_screen_20171227200441.png)

模型最终得分为0.11772，处于当时排行榜10%的位置。

![snap_screen_20171225210400](https://github.com/dafengzai/Udacity-Machine-Learning-Engineer-Nanodegree-Project/tree/master/capstone_project/MarkdownImages\snap_screen_20171225210400.png)

包含的文件中：

Rossemann_project_report.md 为项目的具体报告。

Rossmann_project.ipynb 为特征工程、数据可视化与模型建立训练的实现。
