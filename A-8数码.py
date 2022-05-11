#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import operator
import matplotlib.pyplot as plt
import time
import pandas as pd
import networkx as nx
import sys
#%matplotlib inline


# In[2]:264


num=int(input(("请输入方阵的行/列数：")))
A=list(input('请输入初始状态(输入0-8的任意排列，中间不加空格): '))
B=list((input('请输入目标状态(输入0-8的任意排列，中间不加空格): ')))


# In[3]:


#求逆序数对数
def inverse_num(A):
    a=0
    for i in range(len(A)):
        for j in range(len(A)):
            if A[i]!='0' and A[j]!='0' and i<j and A[i]>A[j]:
                a+=1
    return a


# In[4]:


#由逆序数判断是否有解
def Solve(A,B):
    a=1
    if (inverse_num(A)-inverse_num(B))%2!=0:
        a=0
        print('无解')
    return a


# In[5]:


#将传入元素排成3X3矩阵
z=0
M=np.zeros((num,num),dtype=int)
N=np.zeros((num,num),dtype=int)
for i in range(num):
    for j in range(num):
        M[i][j]=A[z]
        N[i][j]=B[z]
        z+=1


# In[6]:


#定义节点状态
class State:
    def __init__(self,m):
        self.node=m
        self.f=0#f(n)=g(n)+h(n)
        self.g=0
        self.h=0
        self.father=None


# In[7]:


#初始状态与目标状态
init=State(M)
goal=State(N)


# In[8]:


#启发函数h1=不在位的棋子数
def h1(s):
    a=0
    for i in range(len(s.node)):
        for j in range(len(s.node[i])):
            tmp=s.node[i][j]
            for m in range(len(goal.node)):
                for n in range(len(goal.node[m])):
                    if tmp==goal.node[m][n]:
                        a=a+(m-i)**2+(n-j)**2
    return a


# In[9]:


#启发函数h2=所有棋子到其目标位置的距离和
def h2(s):
    a=0
    for i in range(len(s.node)):
        for j in range(len(s.node[i])):
            tmp=s.node[i][j]
            for m in range(len(goal.node)):
                for n in range(len(goal.node[m])):
                    if tmp==goal.node[m][n]:
                        a=a+abs(m-i)+abs(n-j)
    return a


# In[10]:


#对节点按f值排序
def list_sort(l):
    cmp=operator.attrgetter('f')
    l.sort(key=cmp)


# In[11]:


#定义搜索函数，由algorithm关键字确定搜索方法

openlist_contrast=[]
closelist_contrast=[]

def Search(s,algorithm='A*-h1'):
    
    time_start=time.time()
    if algorithm=='DFS':
        print('DFS(maxDepth=8)-防止传统DFS陷入死循环')
    elif algorithm=='BFS':
        print('BFS')
    elif algorithm=='A*-h1':
        print('A*-h1=欧氏距离的平方')
    elif algorithm=='A*-h2=曼哈顿距离':
        print('A*-h2=')
        
    global openlist_contrast
    global closelist_contrast
    global A,B
    if Solve(A,B)==0:
        sys.exit(0)  
        
    openlist_contrast=[s]
    while (openlist_contrast):
        get=openlist_contrast[0]
        if(get.node==goal.node).all():
            openlist_contrast.remove(get)
            closelist_contrast.append(get)
            time_end=time.time()
            print('time cost:',round(1000*(time_end-time_start),3),'ms')
            return get     
        
        openlist_contrast.remove(get)
        closelist_contrast.append(get)
        #判断此时空格位置(a,b)
        
        for a in range(len(get.node)):
            for b in range(len(get.node[a])):
                if get.node[a][b]==0:
                    break
            if get.node[a][b]==0:
                break
               
        #开始移动
        for i in range(len(get.node)):
            for j in range(len(get.node[i])):
                c=get.node.copy()                    
                if (i-a)**2+(j-b)**2==1:                    
                    #直接图进行操作
                    c[a][b]=c[i][j]
                    c[i][j]=0
                    #每个new为一个节点
                    new=State(c)
                    new.father=get
                    new.g=get.g+1#新节点与父亲节点的深度差1
                    
                    if algorithm=='A*-h1':
                        new.h=h1(new)
                    elif algorithm=='A*-h2':
                        new.h=h2(new)
                    else:
                        new.h=0
                    new.f=new.g+new.h
                    #判断是否已经搜索过该节点
                    Add=1
                    if algorithm=='DFS' and new.g>8:
                        Add=0
                    for item in closelist_contrast:
                        if (item.node==new.node).all():
                            Add=0
                    for item in openlist_contrast:
                        if (item.node==new.node).all():
                            if item.f>=new.f:
                                Add=1
                                openlist_contrast.remove(item)
                            elif item.f<new.f:
                                Add=0
                    if Add==1:
                        if algorithm=='DFS':
                            openlist_contrast.insert(0,new)
                        else:
                            openlist_contrast.append(new)
                            
        if algorithm=='A*-h1':
            list_sort(openlist_contrast)
        elif algorithm=='A*-h2':
            list_sort(openlist_contrast)


# In[12]:


#计算有效分支因子b*: N+1 = 1+ b* + pow(b*,2) + pow(b*,3) + … + pow(b*,maxDepth)-N为搜索节点数
def compute_b(N,maxDepth):
    b=1
    sum=0
    while(sum<=N+1):
        sum=0
        for i in range(maxDepth+1):
            sum+=pow(b,i)+0.001
        b+=0.1
    print('有效分支因子b*: ',round(b,1))

directpath=[]
#按照先父后子的顺序进行输出
def printpath(f):
    if f is None:
        return
    printpath(f.father)
    directpath.append(f)
    print(f.node)


# In[14]:


openlist_contrast=[]
closelist_contrast=[]
if Solve(A,B): 
    final=Search(init,algorithm='A*-h1')
    print('搜索得到的最短路径为: ')
    printpath(final)


# In[18]:

pathlist_contrast=openlist_contrast+closelist_contrast
list1=[]
list2=[]
for i in pathlist_contrast:
    for j in pathlist_contrast:
        if j.father==i and j.g<6:
            list1.append(i)
            list2.append(j)
        elif j.father==i and j in directpath:
            list1.append(i)
            list2.append(j)


# Build a dataframe with your connections
df = pd.DataFrame({ 'father':[str(i.node) for i in list1],"son":[str(j.node) for j in list2]})
df
colormap=[]
newlist=[]
for i in range(len(list1)):
    n1=list1[i]
    n2=list2[i]
    addblue=1
    if n1 not in newlist:
        newlist.append(n1)
        for item in directpath:
            if (n1.node==item.node).all():
                addblue=0
        if addblue==1:
            colormap.append('white')
        else:
            if (n1.node == init.node).all() or (n2.node == goal.node).all():
                colormap.append('orange')
            else:
                colormap.append('yellow')
    if n2 not in newlist:
        newlist.append(n2)
        for item in directpath:
            if (n2.node==item.node).all():
                addblue=0
        if addblue==1:
            colormap.append('white')
        else:
            if (n2.node==init.node).all() or (n2.node==goal.node).all():
                colormap.append('orange')
            else:
                colormap.append('yellow')


# In[46]:

plt.cla()
# Build your graph 建立表格
G=nx.from_pandas_edgelist(df, 'father', 'son')
plt.figure(figsize=(min(len(list1),2**16),min(len(list1),2**16)))
# Graph with Custom nodes: 自定义表格
# with_labels是否显示标签，node_size节点大小，node_color节点颜色，node_shape节点形状，alpha透明度，linewidths线条宽度
nx.draw(G, with_labels=True, font_size=max(1*len(list1),20),node_size=250*len(list1), node_color=colormap,node_shape='s', alpha=0.8, linewidths=3*len(list1))
#plt.title('red-shortest path.blue-search node', fontsize=20)

plt.savefig('searchTree.jpg')
print('请打开searchTree.jpg查看A*-h1绘制的搜索树')


# In[44]:


is_contrast=int(input('是否查看(DFS/BFS)算法对比(时间较久），是输入1，否输入0:'))
if is_contrast:
    print('搜索算法对比')
    print('-----------------------------------------------')
    openlist_contrast=[]
    closelist_contrast=[]
    Search(init,algorithm='A*-h1')
    pathlist_contrast=openlist_contrast+closelist_contrast
    print('maxDepth:',pathlist_contrast[-1].g)
    print('搜索节点数目:',len(pathlist_contrast))
    compute_b(len(pathlist_contrast),pathlist_contrast[-1].g)
    print('-----------------------------------------------')
    openlist_contrast=[]
    closelist_contrast=[]
    Search(init,algorithm='A*-h2')
    pathlist_contrast=openlist_contrast+closelist_contrast
    print('maxDepth:',pathlist_contrast[-1].g)
    print('搜索节点数目:',len(pathlist_contrast))
    compute_b(len(pathlist_contrast),pathlist_contrast[-1].g)
    print('-----------------------------------------------') 
    openlist_contrast=[]
    closelist_contrast=[]
    Search(init,algorithm='DFS')
    pathlist_contrast=openlist_contrast+closelist_contrast
    print('maxDepth: 8')
    print('搜索节点数目:',len(pathlist_contrast))
    compute_b(len(pathlist_contrast),8)
    print('-----------------------------------------------')
    openlist_contrast=[]
    closelist_contrast=[]
    Search(init,algorithm='BFS')
    pathlist_contrast=openlist_contrast+closelist_contrast
    print('maxDepth:',pathlist_contrast[-1].g)
    print('搜索节点数目:',len(pathlist_contrast))
    compute_b(len(pathlist_contrast),pathlist_contrast[-1].g)
    print('-----------------------------------------------')


# In[ ]:




