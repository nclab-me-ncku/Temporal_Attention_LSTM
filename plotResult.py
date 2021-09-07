import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

sns.set_theme()
sns.set(font_scale=2)

# df = pd.read_csv('./LSTM_HASH.csv')

# df['session_index'] = df['session_index'].astype(int).astype(str)

# group = df.drop(columns=['session_name']).groupby(['method', 'axis', 'movement']).agg({'r2':'mean'})
# a = [[index[0], row['r2'], index[1], index[2], 'mean', 'indy', '64ms', 'mean'] for index, row in group.iterrows()]
# df = df.append(pd.DataFrame(a, columns=['method', 'r2', 'axis', 'movement', 'session_name', 'monkey', 'binwidth', 'session_index']), ignore_index=True)


# g = sns.catplot(data=df, x='session_index', y='r2', hue='method', col='movement', row='axis', kind='bar', height=5, aspect=4)
# # g.set_xticklabels(rotation=30, ha='left')
# g.set(ylim=[0, 1])

# g.fig.subplots_adjust(top=0.9)
# g.fig.suptitle('monkey indy decode performance')


# # plt.tight_layout()
# # plt.show()
# plt.savefig('./1.jpg')


df = pd.read_csv('./LSTM_monkeyNTimeLag.csv')

filelist = sorted([os.path.splitext(f)[0] for f in os.listdir('./data/monkeyN') if f.endswith('feather')])

df['session_index'] = df['session'].apply(lambda x: filelist.index(x))

g = sns.catplot(data=df, x='session_index', y='r2', hue='trainCount', col='timeLag', col_wrap=1, kind='bar', height=5, aspect=4)
g.set(ylim=[0, 1])

plt.show()
# plt.savefig('./2.jpg')