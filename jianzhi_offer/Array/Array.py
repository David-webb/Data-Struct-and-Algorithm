# -*- coding:utf-8 -*-
import numpy as np

"""
    title: 二维数组中的查找
    在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
    请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
    思路：从左下角的点开始往右上方的方向找
"""
def Array_min(target, array):
    assert isinstance(array,np.ndarray) and array.size
    x,y = array.shape
    tx = x-1
    ty = 0
    while(tx >= 0 and ty <= y-1 ):
        if array[tx][ty] > target:
            tx -= 1
        elif array[tx][ty] < target:
            ty += 1
        else:
            return True
    return False
    pass


"""
    title: 旋转数组的最小数字
    概念：
        数组的旋转：把一个数组最开始的若干个元素搬到数组的末尾。 例如,数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
    要求：
        输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。 
    NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
    思路：最小值正好位于两段中间，用二分查找即可。 
"""

def minNumberInRotateArray(rotateArray):
    start = 0
    end = len(rotateArray) - 1
    while start < end:
        mid = (start + end) // 2
        # 中间值<最左边值，由于本身是有两个递增数组组成，所以中间值右边的根本不需要考虑，肯定都比他大
        # 所以将end移到mid
        if rotateArray[mid] < rotateArray[start]: # 说明最小值在mid左边
            end = mid       # 记不住就考虑{3,4,5,1,2}这种情况就行了,推演的目标就是找到最小值1的位置
        # 中间值>数组start点值，向右移动start点让他是mid
        elif rotateArray[mid] > rotateArray[start]: # 说明最小值在mid右边
            start = mid
        else:
            return rotateArray[mid+1]
    
    pass


"""
    title: 调整数组顺序使奇数位于偶数前⾯面
    输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部
    分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
"""
def reOrderArray(array):
    odd = []
    even = []
    for i in array:
        if i % 2:
            odd.append(i)
        else:
            even.append(i)
    return odd + even
    pass

"""
    title: 数组中出现次数超过一半的数字
    要求：
        数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。
    由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
"""
from collections import Counter
def MoreThanHalfNum_Solution(numbers):
    tt = Counter(numbers)
    for k,v in tt.items():
        if v > (len(numbers) / 2):
            return k
    return 0


"""
    title: 连续子数组的最大和
    HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老
    的⼀一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,
    如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-
    15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。你会不会被他忽悠住？(子向量的长
    度至少是1)
"""
def FindGreatestSumOfSubArray(array):
    if not array:
        return []
    tsum = -0xffffff
    result = sum
    for i in array:
        if sum > 0:
            sum += i
        else:
            sum = i
        result = max(sum, result)
    return result


"""
    title: 把数组排成最小(最大)的数
    要求
        输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
    例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

    思路：
        若 a＋b<b+a a排在在前的规则排序, 如 221 因为 212 < 221 所以 排序后为 212, 实际上sort()
    方法在不传入参数func的时候 默认cmp为None。调⽤用的是lambda x,y: cmp(x, y),而实际上就是调用cmp函数
"""
def cmp_(x,y, mode="min"):
    s1 = str(x) + str(y)
    s2 = str(y) + str(x)
    if s1 <= s2:
        return s1 if mode=="min" else s2
    else:
        return s2 if mode=="min" else s1

def PrintMinNumber(numbers, mode="min"):
    if not len(numbers):
        return []
    flag = True if mode == "max" else False
    numbers.sort(reverse=flag)
    while(len(numbers)>1):
        numbers[0] = cmp_(numbers[0], numbers[1], mode)
        numbers.remove(numbers[1])
    return numbers[0]

"""
   title:数字在排序数组中出现的次数 
   统计一个数字在排序数组中出现的次数。
   思路：最好用二分查找，只要O(logN)
"""
def getNumberOfK(data, k):
    if not data or k not in data:
        return 0

    left = 0
    right = len(data)-1
    firstK = 0
    lastK = 0

    # 确定firstK的位置
    while (left <= right):
        mid = (left + right) // 2
        if data[mid] > k:
            right = mid
        elif data[mid] < k:
            left = mid
        else:
            if mid == 0:
                firstK = 0
                break
            elif data[mid-1] != k:
                firstK = mid
                break
            else:
                right = mid - 1

    # 确定lastK的位置
    left = 0
    right = len(data)-1
    while (left <= right):
        mid = (left + right) // 2
        if data[mid] > k:
            right = mid
        elif data[mid] < k:
            left = mid
        else:
            if mid == (len(data) - 1):
                firstK = len(data)-1
            elif data[mid+1] != k:
                lastK = mid
                break
            else:
                left = mid + 1

    return lastK - firstK + 1
    


"""
    title: 数组中只出现一次的数字
    一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的
    数字。
"""
def findNunsAppearOnce(array):
    l = []
    for n in array:
        if n in l:
            l.remove(n)
        else:
            l.append(n)
    return l
    pass


"""
    title: 数组中重复的数字
    在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有
    几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果
    输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
"""
def duplicate(numbers):
    if numbers == None or numbers == []:
        return 0
    l = []
    ans = []
    for i in numbers:
        if i in l:
            if i not in ans:
                ans.append(i)
        else:
            l.append(i)
    return ans

"""
    title:构建乘积数组(这个有意思，关注下, 这个要结合文档里的图片看)
    给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*…
    *A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
    思路：下三角用连乘可以很容求得，上三角，从下向上也是连乘。因此我们的思路就很清晰了，先
    算下三角中的连乘，即我们先算出B[i]中的一部分，然后倒过来按上三角中的分布规律，把另一部分
    也乘进去。

"""
def multiply(A):
    if not A:
        return []

    # 计算下三角
    leng = len(A)
    B = [1]*leng
    for i in range(1, leng):
        B[i] = B[i-1] * A[i-1]


    # 计算上三角
    tmp = 1
    for i in range(leng-2,-1,-1):
        tmp *= A[i+1]
        B[i] *= tmp

    return B



if __name__ == "__main__":

    # ************ test: Array_min *************
    #tt = np.arange(1, 21).reshape(4,5)
    #print(Array_min(21, tt))

    # ************ test: minNumberInRotateArray ************
    #tt = [2,3,4,5,1]
    #print(minNumberInRotateArray(tt))

    # 
    pass
