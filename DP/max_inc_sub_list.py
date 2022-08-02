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
    

if __name__ == "__main__":
    L = [1,2,5,6,7,8,3]
    print(max_sub_list(L))
    print(max_inc_sub_list(L))
    pass
