import pandas as pd

df = pd.DataFrame({'Gender' : ['男', '女', '男', '男', '男', '男', '女', '女', '女'],
          'name' : ['周杰伦', '蔡依林', '林俊杰', '周杰伦', '林俊杰', '周杰伦', '田馥甄', '蔡依林', '田馥甄'],
          'income' : [4.5, 2.9, 3.8, 3.7, 4.0, 4.1, 1.9, 4.1, 3.2],
         'expenditure' : [1.5, 1.9, 2.8, 1.7, 4.1, 2.5, 1.1, 3.4, 1.2]
         })
df.head(10)


#根据其中一列分组
df_expenditure_mean = df.groupby(['Gender']).mean()
df_expenditure_mean.head(10)


#根据其中一列分组
df_expenditure_mean = df.groupby(['Gender'],sort=True).count()
df_expenditure_mean.head(10)


#根据其中两列分组
df_expenditure_mean = df.groupby(['Gender', 'name']).mean()
df_expenditure_mean.head(10)


#只对其中一列求均值
df_expenditure_mean = df.groupby(['Gender', 'name'])['income'].mean()
df_expenditure_mean.head



