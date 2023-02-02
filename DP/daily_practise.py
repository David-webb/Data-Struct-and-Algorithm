"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2023-01-31 09:34
 * Filename      : daily_practise.py
 * Description   : 动态规划的日常练习
"""
# -*- coding:utf-8 -*-

def linecut(n):
    """钢条切割：给定各个长度钢条的售价和一根指定长度的钢条，求最大收益的切割方案
    Args: 
        n: 指定的钢条长度
    Returns:
        钢条的最大收益和切割方案
    分析：
        状态转移方程 dp(n) = max(dp(n-ni)+pi) n>=ni
        dp(0) = 0, dp(1) = 1
        优化：如果n特别大，那么怎么优化数据结构，减少内存使用?
    说明：
        该函数是自顶向下实现的, 自底向上会更好一些
    """
    sell_table = {
        1:1, 
        2:5,
        3:8,
        4:9,
        5:10,
        6:17,
        7:17,
        8:20,
        9:24,
        10:30
    }
    keys = list(sell_table.keys())
    dp_ops = [[], [1]]
    dp = [-1]*(n+1)
    dp[0] = 0
    dp[1] = 1

    def full_dp(m, dp, ):
        assert m >= 0, "param n must be natural number!"
        # if m = 0:
            # return dp[0]
        if dp[m] != -1:
            return dp[m]
        u_keys = [k for k in keys if k<=m]
        # print(u_keys,m)
        for k in u_keys:
            if k< m:
                dp[k] = full_dp(k, dp)

        dp[m] = max([(dp[m-k]+sell_table[k]) for k in u_keys])
        # print(dp)
        return dp[m]
        pass

    full_dp(n, dp)
    return dp[n]
    pass


def cutline(n):
    """切割钢条，使得利润最大，给出切割方案
       定义： profit[n] 表示长度为n钢条最佳利润
       profit[n] = max(profit[n-i] + cost[i])   0<i<=n
    说明：
        该函数是自底向上实现的
    """
    cost = {
            1:1, 2:5, 3:8, 4:9, 5:10, 6:17, 7:17, 8:20, 9:24, 10:30
    }

    cut_plan = {0:[]} # 用来保存实际的裁剪方案
    for c in cost.keys():
        cut_plan[c] = []

    profit = [0] * (n+1)
    for i in range(1, n+1):
        props = [(profit[i-c] + cost[c], c) for c in cost.keys() if c <= i]
        choices = [c[1] for c in props]
        props = [c[0] for c in props]
        # print(props)
        profit[i] = max(props)
        this_cut = choices[props.index(profit[i])]
        cut_plan[i] = cut_plan[i-this_cut] + [this_cut]
    print(profit)
    print(cut_plan)
    return profit[-1], cut_plan
    pass

def cutline_upgrade(n):
    """对cutline的空间优化版本
    """
    cost = {
        1:1, 2:5, 3:8, 4:9, 5:10, 6:17, 7:17, 8:20, 9:24, 10:30
    }

    cut_plan = {0:[]} # 用来保存实际的裁剪方案
    for c in cost.keys():
        cut_plan[c] = []
    
    keys = list(cost.keys())
    profit = [0] * (len(keys)+1)
    for i in range(1, n+1):
        if i > len(profit): # 插入len(profit)时，需要进行左移，但是在这之前的计算都是正常的
            props = [(profit[11-c] + cost[c], c) for c in cost.keys() if c <= i] # 填入n(n>10)后，profit的索引0对应的是n-10的最佳盈利，索引10对应的是n的最佳盈利，计算n+1时有，(n+1-c) - (n-10) = 11 - c
        else:
            props = [(profit[(i-c)] + cost[c], c) for c in cost.keys() if c <= i]
        choices = [c[1] for c in props] # 最近一次的cut选择
        props = [c[0] for c in props] # cut方案的对应报价
        # print(props)
        # ============= 空间优化 ================ 
        # profit[i] = max(props)
        # =============== 改为：
        if i > len(keys):
            # 更新cut方案报价
            profit.pop(0) # 左移一次buff
            profit.append(max(props)) # 在buff末尾更新最近一次的cut方案报价
            # 更新cut方案对应buff, 丢弃不再使用的部分
            if (i - 10 - 1) in cut_plan:
                cut_plan.pop(i-10-1)         
            pass
        else:
            profit[i] = max(props)
        # =======================================
        this_cut = choices[props.index(max(props))] # 选中的cut方案
        cut_plan[i] = cut_plan[i-this_cut] + [this_cut]
    # print(profit)
    # print(cut_plan)

    pass


def max_inc_sub_list(nums):
    """最长递增子序列的长度
    分析：
        定义以nums[i]结尾的最长子序列长度为dp[i]
        状态转移方程：
        dp[i] = max(dp[j]+1)  0=<j < i 且 nums[j] <= nums[i] (如果要求严格递增，则设置nums[j]<nums[i])
    """
    assert nums, "检测到空数组"
    n = len(nums)
    dp = [0] * n
    dp[0] = 1
    sub_list_buff = [""] * n
    sub_list_buff[0] = [nums[0]]
    for i in range(1, n):
        idx = [j for j in range(i) if nums[j] < nums[i]]
        if idx:
            tmp = [dp[j] for j in idx]
            dp[i] = max(tmp) + 1
            sub_list_buff[i] = sub_list_buff[idx[tmp.index(dp[i]-1)]] + [nums[i]]
        else:
            dp[i] = 1
            sub_list_buff[i] = [nums[i]]
        pass
    return max(dp), sub_list_buff[dp.index(max(dp))]
    pass

if __name__ == "__main__":
    # print(linecut(11))
    # print(cutline(11))
    # cutline_upgrade(115)

    nums = [10,9,2,5,3,7,101,18]
    print(max_inc_sub_list(nums))
    pass


