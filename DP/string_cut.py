"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-05 11:05
 * Filename      : string_cut.py
 * Description   : 
"""

def string_cut(n):
    """钢筋切割：根据不同长度钢筋的价格，计算长度为n的钢筋的最优切法
    状态转移方程：
        opt(n) = max(cost(i) + opt(n-i))
    注意考虑当n大于cost的最大key的情况
    时间复杂度：O(N^2)
    Args:
        n: 钢筋的长度
    """

    cost = {
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
    max_t = max(cost.keys())
    # 如果n超出价格表最大数值
    ask_n = n
    if n > max_t:
        n = max_t

    opt = {
        1:1,
    }
    if n < 1:
        return -1
    elif n == 1:
        return 1
    else:
        for i in range(2, n+1):
            sum_c = cost[i]
            for j in range(1, i):
                sum_c = max(sum_c, cost[j]+opt[i-j])
            opt[i] = sum_c 

    if ask_n > max_t:
        t = int(ask_n / max_t)
        remder = ask_n % max_t
        print(t, remder)
        ans = (t * cost[max_t] + opt[remder]) if remder else t*cost[max_t]  # 这里有一个前提：cost中尺寸越大，价格越高
    else:
        ans = opt[ask_n]
    return opt, ans 

    pass

if __name__ == "__main__":
    print(string_cut(188))
    pass
