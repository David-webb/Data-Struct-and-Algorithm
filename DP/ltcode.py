"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-10 10:53
 * Filename      : ltcode.py
 * Description   : 
"""
import numpy as np


def maxsumarr(L):
    """
        题目：
            给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
        子数组 是数组中的一个连续部分

            示例 1：
            输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
                    [-2, 1, -2, 4, 3, 5, 6, 1, 5]
            输出：6
            解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

            示例 2：
            输入：nums = [1]
            输出：1

            示例 3：
            输入：nums = [5,4,-1,7,8]
            输出：23

         提示：

             1 <= nums.length <= 105
             -104 <= nums[i] <= 104

            进阶：如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的 分治法 求解。
        思路：
            定义dp[i]为以A[i]结尾的最大连续子数组的和
            状态转移方程：dp[i] = max(dp[i-1]+A[i], A[i])
            边界：dp[0] = A[0]

    """
    n = len(L)
    dp = [-1]*n
    dp[0] = L[0]
    rcd = {0: [L[0]]}
    for i in range(1, n):
        # dp[i] = max(dp[i-1]+L[i], L[i])
        if dp[i-1]+L[i] > L[i]:
            rcd[i] = rcd[i-1]+[L[i]]
            dp[i] = dp[i-1]+L[i]
        else:
            rcd[i] = [L[i]]
            dp[i] = L[i]

    print(dp)
    print(rcd)
    return max(dp), rcd[dp.index(max(dp))]
    pass


def maximumProductSubarray(L):
    """乘积最大子数组
    题目：
        给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
        测试用例的答案是一个 32-位 整数。
        子数组 是数组的连续子序列。

                示例 1:

                输入: nums = [2,3,-2,4]
                输出: 6
                解释: 子数组 [2,3] 有最大乘积 6。

                示例 2:

                输入: nums = [-2,0,-1]
                输出: 0
                解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。


                提示:

                        1 <= nums.length <= 2 * 104
                        -10 <= nums[i] <= 10
                        nums 的任何前缀或后缀的乘积都 保证 是一个 32-位 整数
        思路：
                定义dp[i]为以A[i]结尾的最大连续子数组的和
                状态转移方程：dp[i] = max(dp[i-1]*A[i], A[i])
                边界：dp[0] = A[0]
    """
    n = len(L)
    dp = [-1]*n
    dp[0] = L[0]
    rcd = {0: [L[0]]}
    for i in range(1, n):
        # dp[i] = max(dp[i-1]*L[i], L[i])
        if dp[i-1]*L[i] > L[i]:
            rcd[i] = rcd[i-1]+[L[i]]
            dp[i] = dp[i-1]*L[i]
        else:
            rcd[i] = [L[i]]
            dp[i] = L[i]

    print(dp)
    print(rcd)
    return max(dp), rcd[dp.index(max(dp))]
    pass


def isCode(sub):
    """判断sub子串是否是映射表中的规范码
        sub 只包含数字，并且可能包含前导零。
    """
    n = int(sub)
    if sub[0] == "0":  # and len(sub) > 1:
        return False
    elif n < 1 or n > 26:
        return False
    else:
        return True
    pass


def decodeTrait(s):
    """
    题目：
    一条包含字母 A-Z 的消息通过以下映射进行了 编码 ：

    'A' -> "1"
    'B' -> "2"
    ...
    'Z' -> "26"

    要 解码 已编码的消息，所有数字必须基于上述映射的方法，反向映射回字母（可能有多种方法）。例如，"11106" 可以映射为：
    "AAJF" ，将消息分组为 (1 1 10 6)
    "KJF" ，将消息分组为 (11 10 6)
    注意，消息不能分组为  (1 11 06) ，因为 "06" 不能映射为 "F" ，这是由于 "6" 和 "06" 在映射中并不等价。
    给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。
    题目数据保证答案肯定是一个32位的整数。

    示例 1：
    输入：s = "12"
    输出：2
    解释：它可以解码为 "AB"（1 2）或者 "L"（12）。

    示例 2：
    输入：s = "226"
    输出：3
    解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。

    示例 3：
    输入：s = "0"
    输出：0
    解释：没有字符映射到以 0 开头的数字。
    含有 0 的有效映射是 'J' -> "10" 和 'T'-> "20" 。
    由于没有字符，因此没有有效的方法对此进行解码，因为所有数字都需要映射。

    提示：
        1 <= s.length <= 100
        s 只包含数字，并且可能包含前导零。

    思路：
        定义:
            dp[i]表示以前i个字符构成的子串解码的路径个数
            f("sub_str")函数解释子串是否是编码序号，是：返回True， 否：返回False
            dp[i] = (dp[i-1] if f(A[i-1]) else 0)+(dp[i-2] if f(A[i-1]+A[i]) else 0) # A[i]单独成一个数 + A[i-1]A[i]成一个数
        边界：
            dp[0] = f(A[0])

    """
    n = len(s)
    dp = [0]*(n+1)
    dp[0] = 1  # 考虑[] -> [“18”]这种情况，在计算dp[2]的dp[2] += dp[0]时，dp[0]如果为0，会漏掉18这种情况
    dp[1] = 1 if isCode(s[0]) else 0
    for i in range(2, n+1):
        # t = 1 if isCode(s[i-1]) else 0
        dp[i] = dp[i-1] if isCode(s[i-1]) else 0
        if isCode(s[i-2]+s[i-1]):
            dp[i] += dp[i-2]
    print(dp)
    return dp[-1]
    pass


def stockTrade(L):
    """
        题目：
                给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
        你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利
        润。返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

    示例 1：
        输入：[7,1,5,3,6,4]
        输出：5
        解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
             注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
    示例 2：
        输入：prices = [7,6,4,3,1]
        输出：0
        解释：在这种情况下, 没有交易完成, 所以最大利润为 0。

    提示：
        1 <= prices.length <= 105
        0 <= prices[i] <= 104
    思想：
        定义：
            buyin[i] 为前i天的最小股价
            dp[i]为前i天的利润
        状态转移方程：
            dp[i] = A[i] - buyin[i-1]
            buyin[i] = min(buyin[i-1], A[i])
        """
    n = len(L)
    dp = [0] * n
    dp[0] = 0
    buyin = [1e4+1] * n
    buyin[0] = L[0]

    for i in range(1, n):
        dp[i] = L[i] - buyin[i-1]
        buyin[i] = min(buyin[i-1], L[i])
        pass
    print(dp)
    profit = max(dp) if max(dp) > 0 else 0
    return profit

    pass


def robots_traits(row, col):
    """不同路径
        题目：
                一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
                机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
                问总共有多少条不同的路径？
        思路：
                定义dp[i][j]为从左上角到达(i,j)元素的路径总数
                状态转移方程：
                        dp[i][j] = (dp[i-1][j] if i>=1 else 0 )+ (dp[i][j-1] if j>=1 else 0)
                边界：
                        dp[0][0] = 1
    """
    dp = np.zeros((row, col))
    dp[0,0] = 1
    
    for i in range(row):
        for j in range(col):
            if i == j == 0:
                continue
            dp[i,j] = (dp[i-1,j] if i >= 1 else 0) + \
                (dp[i,j-1] if j >= 1 else 0)

    print(dp)
    return dp[row-1, col-1]

    pass

def grid_min_path(grid):
    """
    题目：
        给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
        说明：每次只能向下或者向右移动一步。
    定义：
        dp[i][j]为（i,j）位置最小和路径的数值
    状态转移方程：
        dp[i][j] = min(dp[i-i][j], dp[i][j-1]) + A[i,j]
    边界：
        dp[0][0] = A[0,0]
    """ 
    row, col = len(grid), len(grid[0])
    dp = np.zeros((row, col))
    dp[0,0] = grid[0][0]
    rcd = {(0,0,): [(0,0)]} 
    for i in range(row):
        for j in range(col):
            if i == j == 0:
                continue
            # dp[i,j] = min((dp[i-1,j] if i >= 1 else 0) + \
                # (dp[i,j-1] if j >= 1 else 0) ) + grid[i,j]
            top = dp[i-1,j] if i > 0 else max(grid[i])+1
            left = dp[i,j-1] if j > 0 else max(grid[i-1])+1
            if j == 0 or top < left:
                dp[i,j] = top + grid[i][j]
                rcd[(i,j)] = rcd[(i-1,j)] + [(i,j)]
            if i == 0 or left < top:
                dp[i,j] = left + grid[i][j]
                rcd[(i,j,)] = rcd[(i,j-1)] + [(i,j)]

    print(dp)
    print(rcd[(row-1, col-1)])
    print(dp[row-1, col-1])
    return dp[row-1, col-1]

def max_sub_palindrome(s):
    """
    """
    n = len(s)
    dp = np.zeros((n,n,))
    for i in range(n):
        dp[i,i] = 1
        pass
    for k in range(2, n+1): # k是子串长度
        for i in range(n-k+1):
            j = i + k - 1 
            if s[i] == s[j]:
                # if j-1 < i+1:
                    # if i == j:                    
                        # dp[i,j] = 1
                    # else:
                        # dp[i,j] = 2
                if k == 2 :
                    dp[i,j] = 2
                else:
                    dp[i,j] = dp[i+1,j-1] + 2
            else:
                if j-1 < i+1:                    
                    dp[i,j] = 0 
                else:
                    dp[i,j] = dp[i+1,j-1] 
    print(dp)
    return dp.max()
    pass



if __name__ == "__main__":
    # ============== 和最大连续子数组测试 ===============
    # L = nums = [-2,1,-3,4,-1,2,1,-5,4]
    # L = nums = [5,4,-1,7,8]
    # L = [1]
    # print(maxsumarr(L))

    # ============== 积最大连续子数组测试 ===============
    # L = nums = [2,3,-2,4]
    # L = nums = [-2,0,-1]
    # print(maximumProductSubarray(L))

    # ============== decodeTraits测试 ===============
    # s = "12"
    # s = "226"
    # s = "0"
    # print(decodeTrait(s))

    # ============== StockTrade测试 ===============
    # L = [7,1,5,3,6,4]
    # L = [7,6,4,3,1]
    # print(stockTrade(L))

    # ============== robots_traits测试 ===============
    # m = 3
    # n = 7
    # m = 3
    # n = 2
    # m = 7
    # n = 3
    # m = 3
    # n = 3
    # print(robots_traits(m,n))

    # ============== grid_min_path测试 ===============
    # grid = [[1,3,1],[1,5,1],[4,2,1]]
    # grid = [[1,2,3],[4,5,6]]
    # grid_min_path(grid)

    # ============== 最大回文子串测试 ===============
    print(max_sub_palindrome("ABCDDCEFA"))
    pass
