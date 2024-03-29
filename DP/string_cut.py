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
    分析：自底向上计算时，只需要记得[n-max(cost), n-min(cost)]窗口范围的数据，之前的可以丢弃，所以直接设置滑动窗口即可
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
if __name__ == "__main__":
    print(string_cut(188))
    pass
