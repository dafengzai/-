该项目源自于Kaggle 比赛Rossmann Store Sales 中,Rossmann 是欧洲一家连锁药店，我们需要根据过去各个Rossmann 药妆店每日的销售情况及相关信息情况，来预测Rossmann 未来3个月的销售额。  

本人使用xgboost模型对数据进行预测，本项目使用的库总共有：

```
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
```

在我的笔记本上(5-4210M)训练最终模型的时间约为4个小时，运行代码为：

```
num_boost_round = 4000

print("Train a XGBoost model")
dtrain = xgb.DMatrix(x_train[features], y_train)
dvalid = xgb.DMatrix(x_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=500, feval=rmspe_xg, verbose_eval=True)
```

![snap_screen_20171227200441](https://raw.githubusercontent.com/dafengzai/Udacity-Machine-Learning-Engineer-Nanodegree-Project/master/capstone_project/MarkdownImages//snap_screen_20171227200441.png)

模型最终得分为：

![snap_screen_20171225210400](https://raw.githubusercontent.com/dafengzai/Udacity-Machine-Learning-Engineer-Nanodegree-Project/master/capstone_project/MarkdownImages/snap_screen_20171225210400.png)
得分排在当时排行榜的10%位置。


包含的文件有：

Rossemann_project_report.pdf 项目的报告；

Rossmann_project.ipynb 数据处理与可视化、特征工程、模型建立与实现的代码实现。
