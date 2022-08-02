# -*- coding:utf-8 -*-

"""
    title: 斐波那契额数列
    要求: 
        大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项。n<=39
"""
def Fibonacci_seq(n):
    """
        最简洁，但是效率最低，存在大量重复计算
    """
    if n < 3:
        return 1
    else:
        return Fibonacci_seq(n-1) + Fibonacci_seq(n-2) 

def Fibonacci_seq_v2(n):
    """
        这个效率更高
    """
    assert isinstance(n,int) and n>0 , "输入参数不是正整数"
    if n < 3:
        return 1
    else:
        fst = 1
        sec = 1
        for i in range(3, n+1):
            tmp = sec
            sec += fst
            fst = tmp
        return sec
    pass



"""
    title；跳台阶，矩形覆盖（一样的规律，都是这一项，对应前两项的加和）
    一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
    思路：这一题就是延续斐波那契数列的思路，(注意：斐波那契数列开始是1，1，2不是1，2开始)青
    蛙跳1级台阶有1种跳法，2级台阶有2种跳法，3级台阶时可以从1级台阶跳上来也可以从2级台阶跳
    上来，即等于1级台阶的跳法加2级台阶的跳法因此n级台阶共有n-2级台阶跳法数+n-1级台阶跳法
    数，n=1，sum=1,n=2,sum=2,n=3,sum=3;
"""
def frogJump(n): # 这里n表示台阶总数
    assert isinstance(n,int) and n>0 , "输入参数不是正整数"
    if n < 3:
        return n
    else:
        fst = 1
        sec = 2
        for i in range(3, n+1):
            tmp = sec
            sec += fst
            fst = tmp
        return sec

    pass


"""
    title: 变态跳台阶
    要求：
        一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台
    阶总共有多少种跳法。
    思路：
        每个台阶都有跳与不跳两种情况（除了最后一个台阶），最后一个台阶必须跳。所以共用
    2^(n-1)中情况。
    我的理解：
        这种情况下，青蛙可以一步完成跳跃，也可以选择任意一个或多个（小于等于n-1）中继点完成任务，那么，问题就等同于：
            “共有n bit的内存，其中最后一位是1（对应最后一个台阶必须要跳）， 那么剩下的n-1bit能够表示多少个数”
        答案：2^(n-1)
"""

def superFrog(n):
    assert isinstance(n,int) and n>0 , "输入参数不是正整数"
    return pow(2,n-1)
    pass

if __name__ == "__main__":
    #print(Fibonacci_seq_v2(30))
    print(frogJump(4))
    pass
