#
#
#

import matplotlib.pyplot as plt

f = open("final_w", "r")

step=25
#x = []
#y = []
i = 1
for l in f:
  x,y = (float(s) for s in l.split(" "))
  if i % step == 0:
    print("{:6d}: {:6.4f} {:6.4f}".format(i, x, y))
    plt.plot([x],[y],"bo")
  i = i + 1

plt.axis([-0.05,0.05,-0.05,0.05])
plt.show()

f = open("final_h", "r")

def splitline(l):
  return [float(s) for s in l.split(" ")]

data = [splitline(l) for l in f]
plt.plot(data[0][::step], data[1][::step], "ro")
plt.axis([-0.05, 0.05, 0.05, -0.05])
plt.show()
