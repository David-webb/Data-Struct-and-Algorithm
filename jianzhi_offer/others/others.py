# -*- coding:utf-8 -*-
import numpy as np
import itertools
import collections
"""
    title: 二进制中1的个数
    输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
    思路：
	// 减1的实质：把最右的1借走去减，此位1变0，右边其余0变1
	// 再与它原来的值做位与运算，刚刚变0的1还是0，但是更右边变1的0全又变回0
	// 最终n就会变0结束循环，以上每一次循环都会把一个1变0，所以循环多少次就是有几个1
    我的理解：
        整数很好理解，但是对于负数，虽然负数用补码表示，但是1的补码依然是它本身（正数的补码是自身），因此补码的减法操作还是适用上述规则的
"""
def NumberOf1(n):
    count = 0
    """
        解释一下为什么要用oxffffffff做与：
            因为原来Python2的int类型有32位和64位一说，但到了Python3，当长度超过32位或64位之后，Python3会自动将其转为长整型，长整型理论上没有长度限制。
            所以要做个类型限定（32位）,否则，对于负数情况会无限循环（参考案例：https://blog.csdn.net/u010005281/article/details/79851154）
        
    """
    while n&0xffffffff != 0: 
        count += 1
        n = n & (n-1)
    return count

def NumberOf1_v2(n):
    if n >= 0:
        return bin(n).count('1')
    else:
        return bin(n & 0xffffffff).count('1')



"""
   title: 数值的整数次方
   给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
"""
def power_(base, exponent):
    if base == 0:
        return 0
    if base == 1 or exponent == 0:
        return 1
    ans = 1
    for _ in range(abs(exponent)):
        ans *= base
    if exponent < 0:
        return 1.0 / ans
    else:    
        return ans
    pass

"""
    title: 顺时针打印矩阵
    输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下矩阵：
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
"""
def printMatrix_clockwise(M):
    assert isinstance(M, np.ndarray), "请输入numpy矩阵"
    R, C = M.shape # R，C分别为行号和列号
    total_elements = R * C
     
    res = []
    start = 0 # 轮数（或者叫圈数）
    while(total_elements > len(res)): # 循环终止条件为res元素个数 = 矩阵中元素的个数，每一个循环的操作包括：左右横，上下竖，右左横，下上竖
        endR = R - start - 1  # 每一轮循环，每一列最底端的坐标 
        endC = C - start - 1  # 每一轮循环，每一行最右边的坐标
        for i in range(start, endC+1):  # 左右横
            res.append(M[start, i])
        if endR > start:                # 上下竖
            for j in range(start+1, endR + 1):
                res.append(M[j, endR])
        if endR > start and endC > start:     # 右左横
            for k in range(endC-1, start-1 , -1): # 注意这里start-1是因为python语法限制，使之不能达到上限（下限）
                res.append(M[endR, k])
        if endR > start and (endC - start) > 1: # 下上竖
            for n in range(endR-1, start, -1):
                res.append(M[n, start])
        start += 1
    return res



"""
    title: 字符串的排列
    输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符
    a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
"""
def Permuations_(ss):
    if not ss:
        return []
    else:
        return sorted(list(set(itertools.permutations(ss))))
    pass

def Permutations_v2(ss):
    if not ss:
        return []
    elif len(ss) == 1:
        return [ss]
    else:
        ans = []
        for i in len(ss):
            res = Permutations_v2(ss[:i] + ss[i+1:])
            ans += [ss[i] + j for j in res]
        return sorted(set(ans))
    pass


"""
    title: 最小的K个数
    输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
    思路：
        堆排序
"""
def getleastNumbers(tinput, K):
     
    pass


"""
    title: 整数中1出现的次数（从1到n整数）
    1~13中包含1的数字有1、10、11、12、13因此共出现6次, 求出任意非负整数区间中1出现的次数。
"""
def NumberOf1Between1AndN(n):
    count = 0
    for i in range(1, n):
        for j in str(i):
            if '1' == j:
                count += 1
    return count

    pass


"""
    title: 丑数
    思路：
        把只包含因子2、3和5的数称作丑数（Ugly Number）。例例如6、8都是丑数，但14不是，因为它包含
    因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
"""
def getUglyNumber(index):
    if 0 == index: 
        return 0
    if 1 == index: 
        return 1
    t2, t3, t5 = 0,0,0
    ugly_list = [1]
    for i in range(1, index):
        ugly_list.append(min(ugly_list[t2]*2, ugly_list[t3]*3, ugly_list[t5]*5))
        if ugly_list[i] == ugly_list[t2]*2:
            t2 += 1 
        if ugly_list[i] == ugly_list[t3]*3:
            t3 += 1 
        if ugly_list[i] == ugly_list[t5]*5:
            t5 += 1 
    return ugly_list[-1], ugly_list[index-1]
        


"""
    title:第一个只出现一次的字符
    在一个字符串(1<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置
"""
def FirstNotRepeatingChar(s):
    ord_dict = collections.OrderedDict()
    for i,j in enumerate(s):
        if j in ord_dict.keys():
            ord_dict[i][0] += 1
        else:
            ord_dict[i] = [1, i]
    for k,v in ord_dict.items():
        if v[0] == 1:
            return v[1]
    return None
    pass


"""
    title: 和为S的两个数字
    输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，如果有多对
    数字的和等于S，输出两个数的乘积最小的。
"""
def findnumberswithsum(arr, tsum):
    copule = [tsum-i for i in arr]
    result = [i for i in arr if i in couple]
    try:
        return result[0], result[-1] # 这里如何保证多组结果的情况下乘积最小？答：周长相同，正方形面积最大
    except:
        return []

"""
    title: 和为S的连续正数序列
        小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他⻢马上就写出了正确答案是100。但
    是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到
    另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S
    的连续正数序列? Good Luck!
"""
def findContinuesSeq(tsum):
    if tsum < 3:
        return []
    mid = (tsum + 1) // 2
    fst = 1
    last = 2
    res = []
    Sum = fst + last
    while fst < mid:
        if Sum == tsum:
            res.append(list(range(fst,last+1)))
            last += 1
            Sum += last
        elif Sum < tsum:
            last += 1
            Sum += last
        else:
            Sum -= fst
            fst += 1
    return res
    pass


"""
    title: 翻转单词顺序列
    牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对
    Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am
    I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat
    对一一的翻转这些单词顺序可不在行，你能帮助他么？
"""
def ReverseSentence(s):
    s = s.strip().split()
    s = s[::-1]
    return " ".join(s)

    
"""
    title: 左旋转字符串
    汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个
    指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序
    列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
"""
def LeftRotateString(s, n):
    assert isinstance(s, str), "第一个参数不是字符串类型"
    s = list(s.strip())
    for i in range(n):
        t = s[0]
        s.remove[s[0]]
        s.append(t)
    return "".join(s)
    pass

"""
    title: 扑克牌顺子
    LL今天⼼心情特别好,因为他去买了了⼀一副扑克牌,发现⾥里里⾯面居然有2个⼤大王,2个⼩小王(⼀一副牌原本是54张
    ^_^)...他随机从中抽出了了5张牌,想测测⾃自⼰己的⼿手⽓气,看看能不不能抽到顺⼦子,如果抽到的话,他决定去买体
    育彩票,嘿嘿！！“红⼼心A,⿊黑桃3,⼩小王,⼤大王,⽅方⽚片5”,“Oh My God!”不不是顺⼦子.....LL不不⾼高兴了了,他想了了想,决
    定⼤大\⼩小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上⾯面的5张牌就可以变成
    “1,2,3,4,5”(⼤大⼩小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使⽤用这幅牌模
    拟上⾯面的过程,然后告诉我们LL的运⽓气如何。为了了⽅方便便起⻅见,你可以认为⼤大⼩小王是0。
"""
def IsContinuous(numbers):
    numbers.sort()
    n_of_jokers = len([i for i in numbers if 0 == i])
    dis_sum = 0
    for j in range(n_of_jokers, len(numbers)-1):
        td = numbers[j+1] - numbers[j] - 1
        if -1 == td:
            return False
        else:
            dis_sum += td
    if dis_sum > n_of_jokers:
        return False
    else:
        return True
    pass

"""
    title: 圆圈中最后剩下的数字（约瑟夫环）
    每年年六⼀一⼉儿童节,⽜牛客都会准备⼀一些⼩小礼物去看望孤⼉儿院的⼩小朋友,今年年亦是如此。HF作为⽜牛客的资
    深元⽼老老,⾃自然也准备了了⼀一些⼩小游戏。其中,有个游戏是这样的:⾸首先,让⼩小朋友们围成⼀一个⼤大圈。然后,他
    随机指定⼀一个数m,让编号为0的⼩小朋友开始报数。每次喊到m-1的那个⼩小朋友要出列列唱⾸首歌,然后可以
    在礼品箱中任意的挑选礼物,并且不不再回到圈中,从他的下⼀一个⼩小朋友开始,继续0...m-1报数....这样下
    去....直到剩下最后⼀一个⼩小朋友,可以不不⽤用表演,并且拿到⽜牛客名贵的“名侦探柯南”典藏版(名额有限
    哦!!^_^)。请你试着想下,哪个⼩小朋友会得到这份礼品呢？(注：⼩小朋友的编号是从0到n-1)
"""
def LastRemaining_(n, m):  # n表示总人数，m表示周期
    if n < 0 and not isinstance(n, int):
        return -1
    st = list(range(n))
    start = 0
    while len(st) > 1:
        start = (start + m - 1) % len(st)  # 注意每个循环都是从０开始报数，所以一轮报数ｍ次
        st.pop(start)
    return st[0]
    pass

"""
    title:求1+2+3+...+n
        求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语
    句（A?B:C）。
"""
def Sum_(n):
    # 思路：在Python 中，and 和 or 执行布尔逻辑演算。但是它们并不不返回布尔值，⽽而是返回它们实际
    # 进行比较的值之一。（类似C++里面的&&和||的短路路求值）
    return n and n+Sum_(n-1) # 到达０时，and后面的操作就不执行了



"""
    title: 不用加减乘除做加法
    写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
    这题的答案没看懂，结果是对的，可是原理是什么？？？？？？
"""
def Add_(n1, n2):
    if 0 == n2:
        return n1
    return Add(n1^n2, (n1&n2)<<1) # 左边是异或操作，右边是与操作和左移操作
    pass

"""
    title: 把字符串转换成整数
    将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是
    一个合法的数值则返回0
    输入描述:
        输入一个字符串,包括数字字母符号,可以为空
    输出描述:
        如果是合法的数值表达则返回该数字，否则返回0
"""
def str2int(s):
    if not s:
        return None
    assert isinstance(s,str), "输入类型不是字符串"
    pos_neg_dict = {"+":1, "-":-1}
    s = list(s.strip())
    ans = 0 
    pos_neg_flag = 1
    for i in s:
        if i in pos_neg_dict.keys():
            pos_neg_flag = pos_neg_dict[i]
        else:
            try:
                ans = ans*10 + int(i)
            except: # 存在非0-9的字符
                print(s)
                print("字符串转整数失败")
                return None
    return pos_neg_flag * ans


"""
    title:把字符串串转换成浮点数
"""
def str2float(s):
    Int, decimal = s.strip().split(".")
    Int = str2int(Int)
    decimal = list(decimal.strip())
    dans = 0
    for i, d in enumerate(decimal): 
        try:
            dans += int(d)*pow(10,-(i+1))
        except:
            print("字符转浮点数失败！", s)
            return None
    return Int + dans

    pass

"""
    title: 正则表达式匹配
    请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它
    前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串串的所有字符匹配整个模式。
    例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
"""
def match_(s, pattern):
    import re
    k = re.findall(pattern, s)
    return k



if __name__ == "__main__":
    #print(NumberOf1(-10))
    #print(NumberOf1_v2(-10))
    #M = np.array(list(range(1,26))).reshape(5,5)
    #print(printMatrix_clockwise(M))
    #print(findContinuesSeq(101))
    #print(ReverseSentence("student a am I"))
    #print(str2float("12.345"))
    print(match_("abaca", "a.a"))
    pass
