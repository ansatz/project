import time
import numpy as np
import matplotlib.pyplot as plt

#fig=plt.figure()
#plt.axis([0,1000,0,1])
#
#i=0
#x=list()
#y=list()
#
#plt.ion()
#plt.show()
#
#while i <1000:
#    temp_y=np.random.random()
#    x.append(i)
#    y.append(temp_y)
#    plt.scatter(i,temp_y)
#    i+=1
#    plt.draw()
#    time.sleep(0.05)


import matplotlib.animation as animation

def main():
    numframes = 100
    numpoints = 10
    color_data = np.random.random((numframes, numpoints))
    x, y, c = np.random.random((3, numpoints))

    fig = plt.figure()
    scat = plt.scatter(x, y, c=c, s=100)

    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
                                  fargs=(color_data, scat))
    plt.show()

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,

main()
