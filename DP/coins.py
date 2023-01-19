"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-09 11:54
 * Filename      : coins.py
 * Description   : 
"""
# ========== 版本1：只给出个数 ======================
def coins(Sum):
    """给定1元、3元和5元硬币若干（不限），如何最少的凑出11元（或者Sum元）
    思路：动态规划，状态有两种硬币的面值和总金额，后者是给出的固定值，只和面额有关
    定义dp数组：dp(amount) 凑到amount总金额最少使用的硬币个数，
    状态转移方程：dp(amount) = min(dp(amount - coins(i))) + 1, 其中coins(i)都小于等于amount
    边界：
        dp[amount<0] = -1(人为设置), 
        dp[amount=0]=0
    """

    coins = [1, 3, 5]
    dp = [0] * (Sum+1)
    s = 1
    while s <= Sum:
        prop_coins = [coin for coin in coins if coin <= s]
        # print(s, dp)
        prop_routh = [dp[s-prop_coin] for prop_coin in prop_coins]
        dp[s] = min(prop_routh) + 1 
        s += 1
    print(dp, dp[Sum])
    return dp[Sum] 
    pass

# ============ 版本2：给出了具体的方案(面值和个数) ==============
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
    print(give_changes(11))


