"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2023-02-01 11:49
 * Filename      : subseqence.py
 * Description   : 子序列相关的动态规划问题
"""
import numpy as np

# ============================= 最大子数组和 ===================================
def maxSubArray(nums):
    """最大子数组和：给定一个数组，找出其中和最大的子数组，返回和
    Args:
        nums: 输入的数组
    Returns:
        返回子数组和（提升一下要求，返回子数组）
    分析：
        定义dp[i]为表示以nums[i]结尾的最大子数组和。
        dp[i] = max(nums[i], dp[i-1]+nums[i])
    优化：
        空间优化，我们发现dp[i]的数值之和dp[i-1]相关，所以只需要缓存前者即可。但最终的结果需要max(dp)，所以需要在生成dp数组的时候，同时计算最大值、
        sub_array_s（保存最大子数组的buff）同理。
    """

    n = len(nums)
    assert n>=1, "检测到空数组！"
    dp = [0]*n
    sub_array_s = [""] * n
    sub_array_s[0] = [nums[0]]
    dp[0] = nums[0]
    if n == 1:
        return dp[0]
    for i in range(1, len(nums)):

        # dp[i] = max(dp[i-1]+nums[i], nums[i])
        # =============> 改为：
        if dp[i-1]+nums[i] >= nums[i]:
            dp[i] = dp[i-1] + nums[i]
            sub_array_s[i] = sub_array_s[i-1] + [nums[i]]
        else:
            dp[i] = nums[i]
            sub_array_s[i] = [nums[i]]
    return max(dp), sub_array_s[dp.index(max(dp))]
    pass

# ============================= 最大递增子序列 ===================================
def max_sub_list(L):
    """求L的最大递增子序列长度
    动态规划：
    dp[i] = 1   i = 1
    dp[i] = max(dp[j]) + 1        0 =< j <= i-1

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


def max_inc_sub_list(nums):
    """最长递增子序列的长度(并返回子序列)
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



# ============================= 最大递增子序列的应用: 俄罗斯套娃信封问题 ===================================
def envelops(papers):
    """俄罗斯套娃信封问题
    Args:
        papers：二维数组， e.g.[[5,4], [6,4], [6,7], [2,3]], 每个元素代表一个信封的宽高数据
    Returns:
        能够套信封做多的一组的个数
    规则：
        两个信封的宽或者高相等都不能进行嵌套，必须一方的宽高都大于另一方才可
    思路：
        先对papers按照w进行升序排序，对于w相等的，对h按照降序排序，这样得到一个新的h序列，对该序列求解最大子序列即可
    """
     
    pass



# ============================= 最长公共子序列LCS 及其应用 ===================================
## ===================== 精简版：只计算LCS的长度 =======================
def LCS_s(s1, i, s2, j, memo):
    """计算最长公共子序列，返回长度和序列的索引(分别在两个字符串中)
    """
    n, m = len(s1[i:]), len(s2[j:])

    if not n or not m: # 如果存在空串，那么LCS_s长度为0
        return 0
    
    if memo[i,j] != -1:
        return memo[i,j]

    if s1[i] == s2[j]:
        memo[i,j] = LCS_s(s1, i+1, s2, j+1, memo) + 1
    else:
        memo[i,j] = max(LCS_s(s1, i+1, s2, j, memo), LCS_s(s1, i , s2, j+1, memo))
    
    return memo[i, j]
    pass

def longestCommonSubseq_s(s1, s2):
    """最长公共子序列(LCS_s)
    """
    n, m = len(s1), len(s2)
    memo = np.zeros((n,m)) 
    memo[...] = -1

    lcs_n = LCS_s(s1, 0, s2, 0, memo)
    return lcs_n
    pass

## ===================== 完整版，返回LCS的索引 =======================
def LCS(s1, i, s2, j, memo, idx_d):
    """计算最长公共子序列，返回长度和序列的索引(分别在两个字符串中)
    """
    n, m = len(s1[i:]), len(s2[j:])

    if not n or not m: # 如果存在空串，那么LCS长度为0
        return 0, [[],[]]
    
    if memo[i,j] != -1:
        return memo[i,j], idx_d[(i,j)]

    if s1[i] == s2[j]:
        # memo[i,j] = LCS(s1, i+1, s2, j+1, memo, idx_d) + 1
        lcs_tmp, idx_tmp = LCS(s1, i+1, s2, j+1, memo, idx_d)
        memo[i,j] = lcs_tmp + 1
        idx_d[(i,j)][0].append(i)
        idx_d[(i,j)][0].extend(idx_tmp[0])
        idx_d[(i,j)][1].append(j)
        idx_d[(i,j)][1].extend(idx_tmp[1])
    else:
        # memo[i,j] = max(LCS(s1, i+1, s2, j, memo, idx_d), LCS(s1, i , s2, j+1, memo, idx_d))
        situ1, idx_1 = LCS(s1, i+1, s2, j, memo, idx_d)
        situ2, idx_2 = LCS(s1, i , s2, j+1, memo, idx_d)
        if situ1 > situ2:
            memo[i,j] = situ1
            idx_d[(i,j)][0].extend(idx_1[0])
            idx_d[(i,j)][1].extend(idx_1[1])
        else:
            memo[i,j] = situ2
            idx_d[(i,j)][0].extend(idx_2[0])
            idx_d[(i,j)][1].extend(idx_2[1])
    
    # print(idx_d)

    return memo[i, j], idx_d[(i,j)] 
    pass

def longestCommonSubseq(s1, s2):
    """最长公共子序列(LCS)
    """
    n, m = len(s1), len(s2)
    memo = np.zeros((n,m)) 
    memo[...] = -1
    # 用于保存LCS在s1和s2中索引
    idx_d = {}
    for i in range(n):
        for j in range(m):
            idx_d[(i, j)] = [[], []]

    lcs_n, idx_a = LCS(s1, 0, s2, 0, memo, idx_d)
    return lcs_n, idx_a
    pass

# ===================== LCS自底向上版本 ==========================
def longestCommonSubseq_down2top(s1, s2):
    """最长公共子序列(LCS), 自底向上版本
    分析：
        定义dp[i][j] 为s1[...i]和s2[...j], 状态转移方程：
        if s1[i] == s2[j]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + 1
        注意因为用到了i-1和j-1,所以设置memo的时候需要多一位，也就是存在索引偏移
    优化：
        实际上每次求解dp[i][j]只和dp[i-1][j-1]、dp[i-1][j]和dp[i][j-1]相关, 所以空间上可以优化
    """
    n, m = len(s1), len(s2)
    memo = np.zeros((n+1, m+1)) 
    for i in range(n+1): # 索引0是占位buff
        for j in range(m+1):
            if s1[i] == s2[j]:
                memo[i][j] = memo[i-1][j-1] + 1
            else:
                memo[i][j] = max(mem[i-1][j], memo[i][j-1])

    return memo[n][m] 

    pass


# ===================== LCS 的两个拓展应用 =======================
def minDistance(s1, s2):
    """两个字符串的删除操作:给定两个字符串s1和s2, 找到使得s1和s2相同所需的最小步数，每步可以删除任意一个字符串中的一个字符。
    
    示例：
        输入：“sea", "eat"
        输出：2
        解释：第一步将“sea"变为"ea", 第二步将“eat"变为"ea"
    """
    lcs_n = longestCommonSubseq_s(s1, s2)
    s1_n = len(s1)
    s2_n = len(s2)
    return s1_n + s2_n - 2 * lcs_n
    pass


def minimumDeleteSum(s1, s2):
    """两个字符串的最小ASCII删除和
    """
    lcs_n, idx_n = longestCommonSubseq(s1, s2)
    s1_idx, s2_idx = idx_n
    s1_del = set(range(len(s1))) - set(s1_idx)
    s2_del = set(range(len(s2))) - set(s2_idx)
    ascii_c = [ord(s1[c]) for c in s1_del] + [ord(s2[c]) for c in s2_del]
    tmp_s = sum(ascii_c)
    return tmp_s 
    pass


# ============================== 编辑距离计算（难）===================================
def editDistance(s1, s2):
    """编辑距离计算
    题目：给定两个字符串，计算通过增删改从s1变成s2的最少步骤
    分析：
        定义dp[i][j]为s1[i...]变成s2[j...]最少需要的操作步数

        如果s1[i] == s2[j],则：
            dp[i][j] == dp[i-1][j-1]

        如果s1[i] != s2[j],则：
            dp[i][j] == min(
                    dp[i][j-1], # s1在尾部增加一个字符
                    dp[i-1][j], # s1删除尾部一个字符
                    dp[i-1][j-1], # s1改动一个字符
            ) + 1
        base case:
            如果i = -1, 即s1字符已经遍历完毕, 在s1的开头添加j+1个字符
            如果j = -1, 即s2字符已经遍历完毕, 将s1开头的i+1个字符全部删除

    """
    n, m = len(s1), len(s2)
    memo = np.zeros((n, m))
    memo[...] = -1
    return edit_dp(s1, n-1, s2, m-1, memo)
    pass

def edit_dp(s1, i, s2, j, memo):
    # base case
    if i == -1:
        return j+1
    if j == -1:
        return i+1
    
    if memo[i,j] != -1:
        return memo[i,j]

    if(s1[i] == s2[j]):
        memo[i,j] = edit_dp(s1, i-1, s2, j-1, memo)
    else:
        memo[i,j] = 1 + min(edit_dp(s1, i-1, s2, j, memo), edit_dp(s1, i, s2, j-1, memo), edit_dp(s1, i-1, s2, j-1, memo))
    return memo[i, j]
    pass


# ===================== 编辑距离自底向上版本 ==========================
def editDistance_down2top(s1, s2):
    n, m = len(s1), len(s2)
    memo = np.zeros((n+1, m+1)) 
    memo[...] = 0
    # base case
    for i in range(1, n+1):
        memo[i, 0] = i # 如果s2已经遍历完，后续的操作就是把s1剩下的元素都删除
    for j in range(1, m+1):
        memo[0, j] = j

    for i in range(1, n+1): # 索引0是占位buff
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                memo[i][j] = memo[i-1][j-1]
            else:
                memo[i][j] = min(memo[i-1][j], memo[i][j-1], memo[i-1][j-1]) + 1

    return memo[n][m] 
    pass

# ===================== 编辑距离自底向上版本进阶：添加操作步骤收集 ==========================
def editDistance_d2t_v2(s1, s2):
    n, m = len(s1), len(s2)
    memo = np.zeros((n+1, m+1)) 
    memo[...] = 0
    ops_d = {}
    for i in range(1, n+1):
        for j in range(1, m+1):
            ops_d[(i,j)] = []
            pass

    # base case
    for i in range(1, n+1):
        memo[i, 0] = i # 如果s2已经遍历完，后续的操作就是把s1剩下的元素都删除
        ops_d[(i,0)] = [('del', i-1,)] # 删除s1[i]
    for j in range(1, m+1):
        memo[0, j] = j
        ops_d[(0,j)] = [('insert', s2[j-1],)] # 在s1[i=0]位置添加s2[j]
    ops_d[(0,0)] = ['Start']
    for i in range(1, n+1): # 索引0是占位buff
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                memo[i][j] = memo[i-1][j-1]
                ops_d[(i,j)] = ops_d[(i-1, j-1)] + [('skip',)]
            else:
                # memo[i][j] = min(memo[i-1][j], memo[i][j-1], memo[i-1][j-1]) + 1
                next_op = [memo[i-1][j], memo[i][j-1], memo[i-1][j-1]]
                #更新memo
                memo[i,j] = min(next_op) + 1
                # 更新操作列表
                ops_set = [[("del",i-1,)], [("insert", s2[j-1])], [("replace", i-1, j-1)]] # 删除s1[i], 在当前i后面插入s2[j], 修改s1[i]为s2[j]
                idx_set = [(i-1, j,), (i,j-1,), (i-1, j-1,)]
                choice = next_op.index(min(next_op))
                ops_d[(i,j)] = ops_d[idx_set[choice]] + ops_set[choice]

    return memo[n][m], ops_d[(n,m)] 
    pass

if __name__ == "__main__":
    # 测试最大子数组和 
    nums = [-3,1,3,-1,2,-4,2]
    print(maxSubArray(nums))

    # 测试最大递增子序列
    L = [1,2,5,6,7,8,3]
    print(max_sub_list(L))
    print(max_inc_sub_list(L))

    # 测试LCS
    # s1 = "zabcde"
    # s2 = "acez"
    # print(longestCommonSubseq(s1, s2))
    # print(longestCommonSubseq_s(s1, s2))

    # 测试LCS的应用
    # s1 = "sea"
    # s2 = "eat"
    # print(minimumDeleteSum(s1, s2))

    # 测试编辑距离
    s1 = "horse"
    s2 = "ros"
    print(editDistance(s1, s2))
    print(editDistance_down2top(s1, s2))
    print(editDistance_d2t_v2(s1, s2))
    

    pass
