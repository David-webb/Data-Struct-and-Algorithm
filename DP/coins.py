"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-09 11:54
 * Filename      : coins.py
 * Description   : 
"""


def give_changes(amount):
    """题目：假设有1元，3元，5元的硬币若干（无限），现在需要凑出amount=11元，问如何组合才能使硬币的数量最少
    思路：
        定义dp[amount] 表示凑出amount元需要的硬币总数
        状态转移方程：
            dp[amount] = min(dp[amount - coin[i]]) + 1
    """

    coins = [1,3,5]
    dp = [amount+1] * (amount+1)
    rcd = {} # 记录凑零钱的方案
    for i in range(amount+1):
        rcd[i] = [] 
    dp[0] = 0
    for i in range(1, amount+1):
        for coin in coins:
            if i - coin < 0:
                continue
            else:
                if dp[i-coin]+1 < dp[i]:
                    rcd[i] = rcd[i-coin] + [coin] 
                    dp[i] = dp[i-coin] + 1
                # dp[i] = min(dp[i], dp[i-coin] + 1)
    print(dp)
    print(rcd)
    return dp[amount]

if __name__ == "__main__":
    print(give_changes(10))


