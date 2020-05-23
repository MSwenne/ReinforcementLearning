import matplotlib.pyplot as plt
import sys
import os

f = open(sys.argv[1], 'r')

score1 = []
score2 = []
score3 = []
score4 = []
labels = f.readline()
labels = labels.split(',')
labels[-1] = labels[-1][0:-1]

for line in f.readlines():
    words = line.split(',')
    score1.append(float(words[0]))
    score2.append(float(words[1]))
    score3.append(float(words[2]))
    score4.append(float(words[3][0:-1]))
avg = 50
score11 = []
for i in range(len(score1)-avg):
    score11.append(sum(score1[i:i+avg])/avg)
score21 = []
for i in range(len(score2)-avg):
    score21.append(sum(score2[i:i+avg])/avg)
score31 = []
for i in range(len(score3)-avg):
    score31.append(sum(score3[i:i+avg])/avg)
score41 = []
for i in range(len(score4)-avg):
    score41.append(sum(score4[i:i+avg])/avg)
plt.plot(score1)
plt.plot(score2)
plt.plot(score3)
plt.plot(score4)
plt.legend(labels)

plt.title("Mean of each rating per round")
plt.ylabel('Score')
plt.xlabel('Iterations')
plt.show()

plt.plot(range(avg,len(score1)),score11)
plt.plot(range(avg,len(score1)),score21)
plt.plot(range(avg,len(score1)),score31)
plt.plot(range(avg,len(score1)),score41)
plt.legend(labels)

plt.title(f"Avg over last {avg} rounds")
plt.ylabel('Score')
plt.xlabel('Iterations')
plt.show()