"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2023-01-31 09:34
 * Filename      : daily_practise.py
 * Description   : 动态规划的日常练习
"""
# -*- coding:utf-8 -*-
import numpy as np

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

def longest_palidrome_substring(s):
    """最长回文子串
    """
    n = len(s)
    dp = np.zeros((n,n,)) 
    for i in range(n):
        dp[i,i] = 1
    max_len = 1
    start = 0
    for k in range(2, n+1): 
        for i in range(n-k+1): 
            j = i + k - 1
            if s[i] == s[j]:
                if k == 2:
                    dp[i, j] = 1
                else:
                    dp[i,j] = dp[i+1, j-1]
            if dp[i,j] and k > max_len:
                max_len = k
                start = i

    return s[start:start+max_len] 
    pass

def one_bag():
    """0-1背包
    分析：
        问题：对于装载量为w的bag和i种物品（价值不一），bag的最大价值装载
        状态：bag的容量，物品的种类（重量和价值）
        选择：物品是否放入bag
        定义：dp[i][w]为对于前i种物品和载重为w的bag, 后者可以装载的最大价值
        对于第i件物品，有两种状态:
            放入背包: dp[i-1, w-wt[i]] + val[i-1] # 这里注意dp中的i索引是从1开始，val中是从0开始
            放入背包：dp[i-1, w]
            所以，有dp[i,w] = max(dp[i-1, w], dp[i-1. w-wt[i-1]] + val[i-1])
        边界：
            dp[0,...] = 0
            dp[...,0] = 0
    """
    N = 3 # 物品个数
    W = 4 # 背包的装载量
    wt = [2, 1, 3] # 物体的重量
    val = [4, 2, 3] # 物品的价值
    dp = np.zeros((N+1, W+1))
    for i in range(1, N+1):
        for w in range(1, W+1):
            if w >= wt[i-1]:
                dp[i][w] = max(dp[i-1, w], dp[i-1, w-wt[i-1]] + val[i-1])
                #如果要保存具体的物品列表，在这里判断max的choice索引，然后append对应的物品到buff即可
            else:
                dp[i][w] = dp[i-1, w]
    print(dp)
    return dp[-1,-1]
    pass

def full_bag():
    """完全背包：和0-1背包不同在于，每种物品的个数无限
    分析：
        状态：背包的载重、物品种类
        选择：每种物品当前次装载是否可以装入
        定义：dp[w] 表示载重为w的bag能够装载的物品最大价值, 因为物品无限，所以没有必要在dp table设置对应维度
            dp[w] = max(dp[w-wt[i]] + val[i])  w>=wt[i]
        边界：dp[0] = 0
    """
    N = 3 # 物品个数
    W = 5 # 背包的装载量
    wt = [2, 1, 3] # 物体的重量
    val = [4, 2, 3] # 物品的价值
    dp = np.zeros(W+1)
    for i in range(1, W+1):
        tmp_s = [j for j in range(N) if i>=wt[j]]
        if tmp_s:
            dp[i] = max([(dp[i-wt[j]]+val[j]) for j in tmp_s])
        else:
            dp[i] = dp[i-1]

    print(dp)
    return dp[-1]
    pass

def full_bag_v2():
    """完全背包，按照动态规划的套路来走
    分析：
        状态：背包的载重，物品的种类
        选择：每次选中物品当前是否放入背包
        定义：dp[i][j]表示在背包载重为j和只用前i中物品的情况下，能够装载的最大价值
            dp[i][j] = dp[i-1][j]  # 不放入第i种物品
            dp[i][j] = dp[i][j-wt[i-1]] + val[i-1]  #  放入第i中物品
        边界：dp[0][...] = dp[...][0] = 0
    """
    N = 3 # 物品个数
    W = 5 # 背包的装载量
    wt = [2, 1, 3] # 物体的重量
    val = [4, 2, 3] # 物品的价值
    dp = np.zeros((N+1, W+1))
    for i in range(1, 1+N):
        for j in range(1, 1+W):
            if j < wt[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max((dp[i][j-wt[i-1]] + val[i-1]), dp[i-1][j])
            pass
    print(dp)
    return dp[N,W]
    pass

def findTargetSumWays(nums, target):
    """完全背包的应用：目标和
    题目：输入一个非负整数数组 nums 和一个目标值 target，现在你可以给每一个元素 nums[i] 添加正号 + 或负号 -，
        请你计算有几种符号的组合能够使得 nums 中元素的和为 target
    分析：
        状态：target, 数字的个数
        选择：nums中每个元素是加还是减
        定义：dp[i][j]表示使用前i个数计算和为j的方案个数
        则dp[i][j]可以用两种方式得到:
            dp[i-1][j-nums[i]], 使用前i-1个数计算j-nums[i]的方案，加上nums[i]
            dp[i-1][j+nums[i]], 使用前i-1个数计算j+nums[i]的方案，减去nums[i]
        dp[i][j]是这两种情况的集合，即：
            dp[i][j] = dp[i-1][j-nums[i]] + dp[i-1][j+nums[i]] 
        边界：
            dp[0][...] = 0
        问题: j+nums[i]的上限是target + max(nums), 这样会带来额外的时空开销
    分析2：
        将问题转换成子集背包问题: 将nums分成非负数子集A和负数子集B(B中的元素都是添加-号的),
        则sum(A) - sum(B) = target
        sum(A) = target + sum(B)
        sum(A) + sum(A) = target + sum(B) + sum(A) = target + sum(num) 
        sum(A) = (target + sum(num)) / 2 = T
        从而问题转为子集背包问题, 给定载重量为T的背包和物品列表，物品i的重量为nums[i], 每个物品只用一次，问将背包装满有多少种方案？
        状态：背包载重，物品种类
        选择：物品是否放入背包
        定义dp[i][j]为使用前i个物品装满载重为j的背包的方案个数

        dp[i][j] = dp[i-1][j-nums[i-1]] + dp[i-1][j] # 前者是将i装入背包， 后者是不装入
        边界：dp[0][...] = 0
    """


    pass

if __name__ == "__main__":
    # print(linecut(11))
    # print(cutline(11))
    # cutline_upgrade(115)

    # nums = [10,9,2,5,3,7,101,18]
    # print(max_inc_sub_list(nums))

    # 验证最长回文子串
    # s = "babad"
    # print(longest_palidrome_substring(s))

    # 0-1背包
    # print(one_bag())
    # 完全背包
    full_bag()
    full_bag_v2()
    pass


