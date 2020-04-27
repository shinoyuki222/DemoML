import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
        # Create data
    k = 2
    m = 50
    n = 2
    x1 = np.random.rand(m//2)*4
    x2 = np.random.rand(m//2)*4+4
    x1 = np.hstack([x1,x2])
    y1 = x1 + (np.random.rand(m//2*2)*6-1.5)
    # y1 = np.array([i+np.random.rand()*2-1 for i in x1])
    # y1 = np.random.rand(m)*3#0-3
    datamat = np.vstack([x1,y1]).T
    plt.plot(datamat[:m//2,0],datamat[:m//2,1],'r.')
    plt.plot(datamat[m//2:,0],datamat[m//2:,1],'g.')
    plt.show()


    k = 2
    n = np.shape(datamat)[1]#dims
    meanD = np.mean(datamat,axis=0)
    scale = np.max(datamat,axis=0) - np.min(datamat,axis=0)
    centrs = np.random.uniform(-0.5,0.5,[k,n]) * scale + meanD
    # centrs = np.random.uniform(-0.5,0.5,[k,n])
    print("Initialise matrx\n",centrs)

    plt.plot(datamat[:,0],datamat[:,1],'b.')
    # plt.plot(datamat[:m//2,0],datamat[:m//2,1],'r.')
    # plt.plot(datamat[m//2:,0],datamat[m//2:,1],'g.')
    plt.scatter([centrs[0][0]],[centrs[0][1]],color='',marker='o',edgecolors='red',linewidths=3)
    plt.scatter([centrs[1][0]],[centrs[1][1]],color='',marker='o',edgecolors='green',linewidths=3)
    plt.show()

    for i in range(4):
        dist1 = np.sqrt(np.sum(np.power(np.subtract(datamat,centrs[0]),2),axis=1))
        dist2 = np.sqrt(np.sum(np.power(np.subtract(datamat,centrs[1]),2),axis=1))
        clusterAssment = (dist1 - dist2) >= 0
        Index1 = np.nonzero(clusterAssment==0)[0]
        Index2 = np.nonzero(clusterAssment==1)[0]

        Centr1 = np.mean(datamat[Index1],axis = 0)
        Centr2 = np.mean(datamat[Index2],axis = 0)
        centrs = np.vstack([Centr1,Centr2])
        ax = plt.subplot(2,2,i+1)
        ax.scatter([datamat[Index1,0]],[datamat[Index1,1]],color='',marker='.',edgecolors='red')
        ax.scatter([datamat[Index2,0]],[datamat[Index2,1]],color='',marker='.',edgecolors='green')
        ax.scatter([Centr1[0]],[Centr1[1]],color='',marker='*',edgecolors='red',linewidths=3)
        ax.scatter([Centr2[0]],[Centr2[1]],color='',marker='*',edgecolors='green',linewidths=3)

    plt.show()
