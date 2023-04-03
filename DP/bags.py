"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-09 15:06
 * Filename      : bags.py
 * Description   : 
"""
# -*- coding:utf-8 -*-

import numpy as np
def zero_one_bag(bagW, wl, vl):
    """0-1背包
    题目：给你一个可装载重量为W的背包和N个物品，每个物品有重量和价值两个属性。
    每个物品只能装1次（装或者不装，即0-1的由来）。其中第i个物品的重量为wt[i]，价值为val[i]，现在让你用这个背包装物品，最多能装的价值是多少？
    思路：
        定义dp[i][j]为承重为j的包能够装载前i个物品的总价值
        dp[i][j] = max(dp[i-1][j], dp[i-1][j-wt[i]] + v[i])  如果j-wt[i] >= 0
        dp[i][j] = dp[i-1][j]               如果j-wt[i] < 0
    边界：
        dp[i][0] = dp[0][j] = 0

    Args:
        bagW：表示背包承重的总重量
        wl：物品的重量列表
        vl：物品的价值列表
    """
    n = len(wl)
    assert n == len(vl), "数据wl和vl数量不匹配！"
    dp = np.zeros((n+1,bagW+1), dtype=np.float)
    dp[:,:] = -1

    # 边界
    dp[0,:] = dp[:,0] = 0

    # 
    for i in range(1, n+1):
        for j in range(1, bagW+1):
            if j - wl[i-1] < 0:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1,j], dp[i-1,j-wl[i-1]] + vl[i-1]) # 注意这里wl和vl中使用的索引是i-1

    print(dp)
    return dp.max()
    

def zero_one_bag_inplace(bagW, wl, vl):
    """使用滚动数组代替二维数组，从而降低空间复杂度

        dp[j] = max(dp[j], dp[j-wt[i]] + v[i])  如果j-wt[i] >= 0
        dp[j] = dp[j]               如果j-wt[i] < 0
    边界：
        dp[0] = 0
    注意：要逆向枚举
    Args:
        bagW：表示背包承重的总重量
        wl：物品的重量列表
        vl：物品的价值列表
    """
    n = len(wl)
    assert n == len(vl), "数据wl和vl数量不匹配！"
    dp = np.zeros((bagW+1,), dtype=np.float)
    # dp[:] = -1

    # 边界
    dp[0] = 0
    for i in range(1, n+1):
        for j in range(bagW, 0, -1):
            # if j - wl[i-1] < 0:
                # dp[i][j] = dp[i-1][j]
            # else:
            if (j - wl[i-1]) >= 0:
                dp[j] = max(dp[j], dp[j-wl[i-1]] + vl[i-1]) # 注意这里wl和vl中使用的索引是i-1
    print(dp)
    return dp.max()

def canPartition(nums):
    """输入一个只包含正整数的非空数组nums，判断这个数组是否可以被分割成两个子集，使得两个子集的元素和相等。
    分析：
        转化问题为0-1背包问题，令sum为nums的和，nums个数为N, nums中的数看做物品，每个物品的重量为nums[i]，现存在载重量为sum/2.0的背包，是否存在方案让背包恰好装满
        分析：
            状态：背包的容量，物品的种类
            选择：物品装不装入
            定义：
                dp[i,j]为前i个物品，是否存在将载重为j的背包装满的方案
                dp[i,j] = dp[i-1][j] # 装不下
                dp[i,j] = dp[i-1][j-nums[i-1]] # 装的下，这里的nums[i-1]只是索引偏移
    """
    buff = []
    N = len(nums)
    assert N , "空数组！算个球！"
    if sum(nums) % 2:
        return False
    sum_t = sum(nums) / 2.0
    dp = np.zeros((N+1, sum_t+1,)) # 边界dp[0,...] = dp[...,0] = 0
    for i in range(1, N+1):
        for j in range(1, sum_t+1):
            if dp[i-1,j-nums[i-1]] + nums[i-1] > sum_t: # 超过容量
                dp[i,j] = dp[i-1, j]
                pass
            else:
                dp[i,j] = max(dp[i-1,j], dp[i-1,j-nums[i-1]] + nums[i-1]) 
            if dp[i,j] == sum_t:
                return True
            pass
    return False
    pass

def full_bag(bagW, wl, vl):
    """完全背包：在0-1背包的基础上,增加了条件：每种物品的个数不限，可重复装载
    思路：
        定义dp[i][j]为承重为j的包能够装载前i"种"物品的总价值
        dp[i][j] = max(dp[i-1][j], dp[i][j-wt[i]] + v[i])  如果j-wt[i] >= 0
        dp[i][j] = dp[i-1][j]               如果j-wt[i] < 0
    边界：
        dp[i][0] = dp[0][j] = 0
    Args:
        bagW：表示背包承重的总重量
        wl：物品的重量列表
        vl：物品的价值列表
    """
    n = len(wl)
    assert n == len(vl), "数据wl和vl数量不匹配！"
    dp = np.zeros((n+1,bagW+1), dtype=np.float)
    dp[:,:] = -1

    # 边界
    dp[0,:] = dp[:,0] = 0

    # 
    for i in range(1, n+1):
        for j in range(1, bagW+1):
            if j - wl[i-1] < 0:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1,j], dp[i,j-wl[i-1]] + vl[i-1]) # 注意这里wl和vl中使用的索引是i-1

    print(dp)
    return dp.max()

    pass

def full_bag_inplace(bagW, wl, vl):
    """滚动数组优化
    思路：
        定义dp[i][j]为承重为j的包能够装载前i"种"物品的总价值
        dp[i][j] = max(dp[i-1][j], dp[i][j-wt[i]] + v[i])  如果j-wt[i] >= 0
        dp[i][j] = dp[i-1][j]               如果j-wt[i] < 0
        ====> dp[i][j]只和dp[i-1]和dp[i]的状态有关，所以这里通过正向枚举的方式用一维滚动数组代替二维数组
        dp[j] = max(dp[j], dp[j-wt[i]] + v[i])  如果j-wt[i] >= 0 # 需要正向枚举
        dp[i][j] = dp[i-1][j]               如果j-wt[i] < 0
    边界：
        dp[i][0] = dp[0][j] = 0
    注意：要正向枚举    
    Args:
        bagW：表示背包承重的总重量
        wl：物品的重量列表
        vl：物品的价值列表
    """
    n = len(wl)
    assert n == len(vl), "数据wl和vl数量不匹配！"
    dp = np.zeros((bagW+1,), dtype=np.float)
    # dp[:] = -1

    # 边界
    dp[0] = 0
    for i in range(1, n+1):
        for j in range(1, bagW+1):
            if (j - wl[i-1]) >= 0:
                dp[j] = max(dp[j], dp[j-wl[i-1]] + vl[i-1]) # 注意这里wl和vl中使用的索引是i-1
    print(dp)
    return dp.max()

def bounded_knapsack(bagW, wl, vl, nl):
    """多背包问题
    题目：多重背包（bounded knapsack problem）与前面不同就是每种物品是有限个：一共有N种物品，第i（i从1开始）种物品的i
    数量为n[i]，重量为w[i]，价值为v[i]。在总重量不超过背包承载上限W的情况下，能够装入背包的最大价值是多少？
    思路：
        定义dp[i][j]为承重为j的包能够装载前i"种"物品的总价值
        dp[i][j] = max(dp[i-1][j], dp[i][j-k*wt[i]] + k*v[i])  如果j-k*wt[i] >= 0     0=<k<=n[i]
        dp[i][j] = dp[i-1][j]               如果j-k*wt[i] < 0
    边界：
        dp[i][0] = dp[0][j] = 0
    """

    n = len(wl)
    assert n == len(vl), "数据wl和vl数量不匹配！"
    dp = np.zeros((n+1,bagW+1), dtype=np.float)
    dp[:,:] = -1

    # 边界
    dp[0,:] = dp[:,0] = 0

    # 
    for i in range(1, n+1):
        for j in range(1, bagW+1):
            bound = min(nl[i-1], int(j/wl[i-1]))
            for k in range(0, bound+1):
                # (deprecated) bound已经做了约束，保证k*wl[i-1]不会超出j的限制
                # if j - k*wl[i-1] < 0: 
                    # dp[i][j] = max(dp[i-1][j], dp[i][j])
                # else:
                    # dp[i][j] = max(dp[i-1,j], dp[i,j-k*wl[i-1]] + k*vl[i-1]) # 注意这里wl和vl中使用的索引是i-1
                dp[i][j] = max(dp[i-1,j], dp[i,j-k*wl[i-1]] + k*vl[i-1]) # 注意这里wl和vl中使用的索引是i-1

    print(dp)
    return dp.max()
    pass

def bounded_knapsack_inplace(bagW, wl, vl, nl):
    """多背包问题
    题目：多重背包（bounded knapsack problem）与前面不同就是每种物品是有限个：一共有N种物品，第i（i从1开始）种物品的i
    数量为n[i]，重量为w[i]，价值为v[i]。在总重量不超过背包承载上限W的情况下，能够装入背包的最大价值是多少？
    思路：
        定义dp[i][j]为承重为j的包能够装载前i"种"物品的总价值
        dp[i][j] = max(dp[i-1][j], dp[i][j-k*wt[i]] + k*v[i])  如果j-k*wt[i] >= 0     0=<k<=n[i]
        dp[i][j] = dp[i-1][j]               如果j-k*wt[i] < 0
    边界：
    """

    n = len(wl)
    assert n == len(vl), "数据wl和vl数量不匹配！"
    dp = np.zeros((n+1,bagW+1), dtype=np.float)
    dp[:,:] = -1

    # 边界
    dp[0,:] = dp[:,0] = 0

    # 
    for i in range(1, n+1):
        for j in range(1, bagW+1):
            bound = min(nl[i-1], int(j/wl[i-1]))
            for k in range(0, bound+1):
                # (deprecated) bound已经做了约束，保证k*wl[i-1]不会超出j的限制
                # if j - k*wl[i-1] < 0: 
                    # dp[i][j] = max(dp[i-1][j], dp[i][j])
                # else:
                    # dp[i][j] = max(dp[i-1,j], dp[i,j-k*wl[i-1]] + k*vl[i-1]) # 注意这里wl和vl中使用的索引是i-1
                dp[i][j] = max(dp[i-1,j], dp[i,j-k*wl[i-1]] + k*vl[i-1]) # 注意这里wl和vl中使用的索引是i-1

    print(dp)
    return dp.max()
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

def DP_recollect():
    pass

if __name__ == "__main__":
    bagW = 20
    wl = [5, 3, 7, 9]
    vl = [3, 10, 8, 4]
    # print(zero_one_bag(bagW, wl, vl))
    # print(zero_one_bag_inplace(bagW, wl, vl))
    # print(full_bag(bagW, wl, vl))
    # print(full_bag_inplace(bagW, wl, vl))
    nl = [3,5,6,9]
    bounded_knapsack(bagW, wl, vl, nl)
    pass
