import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

f = open('./data.txt','r')
data =  f.readlines()

WIN_DATA = data[0].split()
SCORE_DATA = data[1].split()
for i, data in enumerate(WIN_DATA):
    if i == 0:
        WIN_DATA[i] = int(data[1:-1] == 'True')
    else:
        WIN_DATA[i] = int(data[:-1] == 'True')
for i, data in enumerate(SCORE_DATA):
    if i == 0:
        SCORE_DATA[i] = float(data[1:-1])
    else:
        SCORE_DATA[i] = float(data[:-1])

avg_win = [np.mean(WIN_DATA[i:i+50]) for i in range(len(WIN_DATA)-50)]
avg_score = [np.mean(SCORE_DATA[i:i+50]) for i in range(len(SCORE_DATA)-50)]
lose = [i for i, x in enumerate(WIN_DATA) if not x]
win = [i for i, x in enumerate(WIN_DATA) if x]
score_lose = [i for i, x in enumerate(SCORE_DATA) if x <= -199]
score_win = [i for i, x in enumerate(SCORE_DATA) if x > -199]
fig,a =  plt.subplots(2,1,figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
fig.suptitle('Mountain Car DQN training', size=14)
a[0].scatter(score_lose,[SCORE_DATA[i] for i in score_lose] , c='r', s=4)
a[0].scatter(score_win,[SCORE_DATA[i] for i in score_win] , c='g', s=4)
a[0].plot(range(50,len(SCORE_DATA)), avg_score, c='b', label='mean of last 50 games', markersize=4)
a[0].set_title('Mountain Car DQN scores', size=12)
a[1].scatter(lose,[WIN_DATA[i] for i in lose] , c='r', s=4)
a[1].scatter(win,[WIN_DATA[i] for i in win] , c='g', s=4)
a[1].plot(range(50,len(WIN_DATA)), avg_win, c='b', label='mean', markersize=4)
a[1].set_title('Mountain Car DQN wins', size=12)
plt.sca(a[0])
plt.xticks(range(0,len(WIN_DATA)+1,int(len(WIN_DATA)/11)),range(0,len(WIN_DATA)+1,int(len(WIN_DATA)/11)))
plt.grid(True)
plt.yticks([0, -50, -100, -150, -200], ['0','-50','-100','-150','-200'])
plt.legend(framealpha=1, frameon=True, loc='upper right',prop={'size': 12})
plt.ylabel('Score')
plt.sca(a[1])
plt.grid(True)
plt.xticks(range(0,len(WIN_DATA)+1,int(len(WIN_DATA)/11)),range(0,len(WIN_DATA)+1,int(len(WIN_DATA)/11)))
plt.yticks([1.0, 0.75, 0.5, 0.25, 0.0], ['100%','75%','50%','25%','0%'])
plt.ylabel('Win')
fig.tight_layout(pad=4)
plt.savefig('stats.png')
plt.show()
