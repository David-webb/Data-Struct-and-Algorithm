# -*- coding:utf-8 -*-

# class Solution {
    # public int lengthOfLIS(int[] nums) {
        # if (nums.length == ) {
            # return ;
        # }
        # int[] dp = new int[nums.length];
        # //初始化就是边界情况
        # dp[] = 1;
        # int maxans = 1;
        # //自底向上遍历
        # for (int i = 1; i < nums.length; i++) {
            # dp[i] = 1;
            # //从下标到i遍历
            # for (int j = ; j < i; j++) {
                # //找到前面比nums[i]小的数nums[j],即有dp[i]= dp[j]+1
                # if (nums[j] < nums[i]) {
                    # //因为会有多个小于nums[i]的数，也就是会存在多种组合了嘛，我们就取最大放到dp[i]
                    # dp[i] = Math.max(dp[i], dp[j] + 1);
                # }
            # }
            # //求出dp[i]后，dp最大那个就是nums的最长递增子序列啦
            # maxans = Math.max(maxans, dp[i]);
        # }
        # return maxans;
    # }
# }


def max_sub_list(L):
    """求L的最大递增子序列长度
    动态规划：
    dp[i] = 1   i = 1
    dp[i] = max(dp[j]) + 1         0 =< j <= i-1

    """
    lenth = len(L)
    if(lenth == 1):
        return 1
    dp = [0] * (lenth)
    dp[0] = 1
    maxans = 1
    for i in range(1, lenth):
        dp[i] = 1
        for j in range(i):
            if L[j] < L[i]:
                dp[i] = max(dp[i], dp[j]+1)
        maxans = max(maxans, dp[i])
    return maxans


def max_inc_sub_list(L):
    """求L的最大递增子序列长度
        暴力破解: 统计以L中每个元素开头的序列集合，取最大的序列长度作为输出
    """
    lenth = len(L)
    ans_l = []
    for i in range(lenth):
        tmp = [L[i]]
        for j in range(i, lenth):
            if L[j] > tmp[-1]:
                tmp.append(L[j])
        ans_l.append(len(tmp))
    return max(ans_l)
    
def longestIncSubseq(nums):
    """给定一个长度为N的整数数组，求其中最长的递增子序列的长度
    注意：这里的的递增子序列不需要是连续的
    思路：判断是否是动态规划问题：
        穷举：可以。以每个元素为起点遍历整个数组。
        重叠子问题：定义dp[i] = n, 表示以nums[i]为起点的最大递增序列长度为n.
            dp[i] = max(dp[j]) + 1 if num[j] >= num[i] else 0, 其中j in [i+1, n] 
        子问题相互独立：满足
        边界：dp[n] = 1
    params:
        nums是整数数组
    """

    l = len(nums)
    dp = [0] * l 
    dp[-1] = 1
    for i in range(l)[-2::-1]:
        prop_j = [dp[j] for j in range(i+1, l) if nums[j]>=nums[i]]
        if prop_j:
            dp[i] = max(prop_j) + 1 
        else:
            dp[i] = 0
    print(dp, max(dp))
    return max(dp) 
    pass


if __name__ == "__main__":
    L = [1,2,5,6,7,8,3]
    print(max_sub_list(L))
    print(max_inc_sub_list(L))
    pass
