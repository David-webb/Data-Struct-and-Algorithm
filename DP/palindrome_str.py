"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-06 17:53
 * Filename      : palindrome_str.py
 * Description   : 回文串相关的几种动态规划区间模型
"""
#-*- coding:utf-8 -*-

import numpy as np


def palindrome(str_m):
    """
        题目: 给定一个长度为n（n <= 1000）的字符串A，求插入最少多少个字符使得它变成一个回文串。
        思路：
        定义dp[i][j]为将子串A[i...j]填充成回文串所需的最小操作次数,
        最优子结构：
            dp[i][j] 可以看做将A[i+1,....,j]填充成回文串的基础上，在左侧添加了A[i]构成了A', 这时候只要在A'右侧添加A[i]即可, 故：dp[i][j] = dp[i+1][j] + 1
            同理，可以有：dp[i][j] = dp[i][j-1] + 1 
                或 dp[i][j] = dp[i+1][j-1] + 2  （但这个已经包含在前面两种情况中了）
            综上：
                dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1 

            另外，还有一种特殊情况，如果子串A[i...j]中的A[i] == A[j], 那么只要将子串A[i+1....j-1]填充成回文串即可，即：
                dp[i][j] = dp[i+1][j-1]
    """
    assert isinstance(str_m, str), "str_m 必须是字符串类型." 
    length = len(str_m)
    if length <= 1:
        return 0

    dp = np.zeros((length, length,))
    # 方法一（可读性更好）：观察状态转移方程，发现需要先求解最小长度的子串的构建所需的步骤数
    # 下面两个for循环的位置不能互换，因为子串A[i...j]的dp数值求解是根据其内部更小的子串构成的，所以这里应当先计算小跨度(k)的dp值
    for k in range(2, length+1):# k表示间隔的长度 
        for i in range(length-k+1):
            j = i + k - 1
            if str_m[i] == str_m[j]:
                if k == 2:
                    dp[i,j] = 0
                else:
                    dp[i,j] = dp[i+1, j-1]
            else:
                dp[i,j] = min(dp[i+1, j], dp[i, j-1]) + 1
        print("============= 当前的步长为：%d =============" % k)
        print(dp)

    # return dp[0, length-1]
    return dp

    # 方法二：思考dp table的递推过程,化图
    # for i in range(length-2, -1, -1):
        # for j in range(i, length):
            # if i < j:
                # if str_m[i] == str_m[j]:
                    # dp[i][j] = dp[i+1][j-1]
                # else:
                    # dp[i][j] = min(dp[i+1,j], dp[i][j-1]) + 1
    # return dp
    # return dp[0][n-1]; 


def the_longest_palindrome_substring(str_m):
    """提取最长的回文子串
       思路：
       定义：dp[i][j]表示子串A[i...j]是回文字符串（1：是，0：否）
       状态转移方程：
            dp[i][j] = dp[i+1][j-1]    A[i] == A[j]
            dp[i][j] = 0               A[i] != A[j]
    """
    assert isinstance(str_m, str), "str_m 必须是字符串类型." 
    length = len(str_m)
    dp = np.zeros((length, length,))
    max_len = 1
    start = 0
    for i in range(length):
        dp[i,i] = 1
    for k in range(2, length+1):
        for i in range(length-k+1):
            j = i+k-1
            if str_m[i] == str_m[j]:
                if k == 2:
                    dp[i,j] = 1
                else:
                    dp[i,j] = dp[i+1, j-1]
            if dp[i,j] and k > max_len:
                max_len = k
                start = i 

    return str_m[start:start+max_len]
    # return dp

    pass

def the_longest_palindrome_subsequence(str_m):
    """提取最长的回文子序列（的长度）
       思路：
       定义dp[i][j]为以i为子串A[i...j]的最长回文子序列的长度
       状态转移方程：
            dp[i][j] = dp[i+1][j-1] + 2         A[i] == A[j]
            dp[i][j] = max(dp[i+1,j], dp[i, j-1])   A[i] != A[j]
       边界：
            dp[i][i] = 1
    """

    assert isinstance(str_m, str), "str_m 必须是字符串类型." 
    length = len(str_m)
    dp = np.zeros((length, length,))
    # tmp_str = np.zeros((length, length,), dtype=str)
    tmp_str = [""] * (length * length)
    # max_len = 1
    # start = 0
    for i in range(length):
        dp[i,i] = 1
        tmp_str[i*length+i] = str_m[i]
    for k in range(2, length+1):
        for i in range(length-k+1):
            j = i+k-1
            if str_m[i] == str_m[j]:
                if k == 2:
                    dp[i,j] = 2
                    tmp_str[i*length+j] = str_m[i]+str_m[j] 
                else:
                    dp[i,j] = dp[i+1, j-1] + 2
                    tmp_str[i*length+j] = str_m[i]+tmp_str[(i+1)*length+j-1] + str_m[j] 
            else:
                dp[i,j] = max(dp[i+1, j], dp[i, j-1])
                tmp_str[i*length+j] = tmp_str[(i+1)*length+j] if dp[i+1,j] > dp[i, j-1] else tmp_str[i*length+j-1]
            # if dp[i,j] and k > max_len:
                # max_len = k
                # start = i 

    # return str_m[start:start+max_len]
    # return dp[0, length-1]
    # return dp
    return dp[0, length-1], tmp_str[0*length+length-1]

    pass


if __name__ == "__main__":
    # print(palindrome("abc,bad")) # abc,bad ===> dabc,cbad
    print(the_longest_palindrome_substring("ABCDDCEFA"))
    # print(the_longest_palindrome_subsequence("ABCDBDCEFA"))

    pass
