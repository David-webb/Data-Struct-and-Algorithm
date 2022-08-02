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
