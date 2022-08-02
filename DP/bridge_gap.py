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

def bridge_gap(L):
    """小朋友过桥
        一个月黑风高的夜晚，有N个小朋友要过桥，每个人通过的时间不一样，表T记录了所有人过桥的时间，
        T[i]表示第i个人的过桥时间。桥上一次只能通过不超过两个人，且大家只有一个手电筒，每次通过后
        需要有人将手电带回来，通过的时间按照两人中最长的算。问所有人最短的通过时间需要多久？
    """
    nums = len(L)
    L.sort()
    ans = [0]*nums
    ans[0] = L[0]
    ans[1] = L[1]
    # ans[2] = L[2]
    for i in range(2, nums):
        ans[i] = min(ans[i-1] + L[0] + L[i], ans[i-2] +L[0]+ L[i] + 2*L[1])
    # print(L)
    # print(ans)
    return ans[-1]
    
    pass

if __name__ == "__main__":
    L = [1,2,5,10,11]
    # L = [1]
    print(bridge_gap(L))
