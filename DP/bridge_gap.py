#-*- coding:utf-8 -*-

# #include<iostream>
# #include<vector>
# #include<algorithm>
# using namespace std;
 
# int main()
# {
   # int n;
	# cin>>n;
	# int *t = new int[n];
	# for (int i=0;i<n;i++)
	# {
		# cin>>t[i];
	# }
	# sort(t,t+n);
	# vector<int> vect(n);
	# vect[0]=0;
	# vect[1]=t[1];
	# for (int i=2;i<n;i++)
	# {
		# vect[i] = min(vect[i-1]+t[0]+t[i],vect[i-2]+t[0]+t[i]+2*t[1]);
	# }
	# cout<<vect[n-1];
 
	# return 0;
# }
from copy import deepcopy

def bridge_gap(L):
    """小朋友过桥
        一个月黑风高的夜晚，有N个小朋友要过桥，每个人通过的时间不一样，表T记录了所有人过桥的时间，
        T[i]表示第i个人的过桥时间。桥上一次只能通过不超过两个人，且大家只有一个手电筒，每次通过后
        需要有人将手电带回来，通过的时间按照两人中最长的算。问所有人最短的通过时间需要多久？
    """
    nums = len(L)
    L.sort()
    print(L)
    ans = [0]*nums
    ans[0] = L[0]
    ans[1] = L[1]
    # ans[2] = L[2]
    for i in range(2, nums):
        ans[i] = min(ans[i-1] + L[0] + L[i], ans[i-2] +L[0]+ L[i] + 2*L[1])
    # print(L)
    print(ans)
    # print(cost)
    return ans[-1]
    
    pass

def crossBridge(T):
    """小朋友过桥:
    一个月黑风高的夜晚，有N个小朋友要过桥，每个人通过的时间不一样，表T记录了所有人过桥的时间，T[i]表示第i个人的过桥时间。桥上一次只能通过不超过两个人，且大家只有一个手电筒，每次通过后需要有人将手电带回来，通过的时间按照两人中最长的算。问所有人最短的通过时间需要多久？
    注意：每次通过后，送手电回来的可以是先前过桥的人！
    初步方案：对T进行排序，耗时最短的两人为s1,s2，那么一轮过桥如下定义：这二人先过桥，s1返回，接着剩下的人中耗时最长的两人过桥，s2返回. 耗时：t = s2 + s1 + max(A,B) + s2 = s2 + s1 + B + s2 (假设B>A)
    从上面看出，最后一轮一定是s1 s2一起过桥。问题在倒数第二轮：
       如果剩下单数人（s1 s2 和一个A）， s1 + A 先过，s1返回；耗时：A + s1
       如果剩下偶数人：和标准程序一样，耗时：t

    在保证s1s2存在的基础上，添加A<B<C过桥，两种情况：
    1. (A,B),C   耗时: t1 = 2*s2 + s1 + B + C + s1
    2. A,(B,C)   耗时: t2 = A + s1 + 2*s2 + s1 + C, 
    t2 - t1 = A - B < 0, 所以，最节省的肯定是A, (B, C)

    综上，
        cost[0] = 0 ; cost[1] = s1 ; cost[2] = s2
        cost[m] = cost[m-1] + T[m-1] + s1   m = 3
        cost[m] = cost[m-2] + T[m] + 2*s2 + s1  m > 3
        即：
        cost[m] = min(cost[m-1] + T[m-1] + s1, cost[m-2] + T[m] + 2*s2 + s1)

    是否是动态规划：
        穷举：满足，可以穷举所有组合情况
        重叠子问题：cost[m] 表示m个人过桥的最短时间, 这里有个前提，对T进行升序排序，然后从前往后加人的
            cost[m] = cost[m-1] + (s1 + T[m]) if m为偶数且m=2 
            cost[m] = cost[m-2] + 2*s2 + T[m] + s1 if m为奇数 
        子问题相互独立：没毛病
        边界：cost[0] = 0 ; cost[1] = s1 ; cost[2] = s2
    """

    T.sort()
    l = len(T)
    cost = [0] * (l+1)
    cost[1] = T[0]
    cost[2] = T[1]
    
    for i in range(3,l+1):
        if i == 3: # m+1为奇数，m为偶数
            cost[i] = cost[i-1] + T[0] + T[i-1] # 这里使用的是T[i-1],因为cost和T的索引上错位gap是1，cost比T多一个元素
        else:
            cost[i] = cost[i-2] + 2*T[1] + T[i-1] + T[0] 
    print(cost) 
    return cost[-1]


if __name__ == "__main__":
    L = [3,1,2,5,10,11, 34, 56]
    # L = [1]
    print(bridge_gap(L))
    print(crossBridge(L))
