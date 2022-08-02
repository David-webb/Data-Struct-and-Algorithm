"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-17 11:23
 * Filename      : sort_alg_set.py
 * Description   : 
    十大排序算法的python实现
        code和算法思想动态图参考:https://www.jianshu.com/p/bbbab7fa77a2
        详细的算法介绍参考:https://www.cnblogs.com/Camilo/p/3929059.html

"""
# -*- coding: utf-8 -*-
from random import randint, randrange

def comp_fn(a, b, mode=1):
    "升序mode==1 降序mode==0"
    return a>b if mode==1 else a<b 

def valid_fn(func):
    def wrapper(*args, **kwargs):
        nums = args[0]
        #assert nums, "数组为空！"
        if len(nums) == 0:
            print("数组为空！")
            #return nums
        func(*args, **kwargs)
    return wrapper
    pass

@valid_fn
def Bubble_sort(numlist, mode=1):
    """
        numlist: 非空数组
        mode: 1,升序； 0,降序
    """
    leng = len(numlist)
    for i in range(leng-1):
        for j in range(i+1, leng):
            if comp_fn(numlist[i], numlist[j], mode=mode):
                #tt = numlist[i]
                #numlist[i] = numlist[j]
                #numlist[j] = tt
                numlist[i], numlist[j] = numlist[j], numlist[i] # Python 交换两个数不用中间变量 
    pass


@valid_fn
def selectionSort(numlist, mode=1):
    """
        选择排序，每次选择剩余未排序部分的最小值与当前位置的值互换
    """
  
    leng = len(numlist)
    
    for i in range(leng-1):
        s_i = i
        for j in range(i+1, leng):
            if comp_fn(numlist[s_i], numlist[j], mode=mode):
                s_i = j
        numlist[i], numlist[s_i] = numlist[s_i], numlist[i] # Python 交换两个数不用中间变量 


@valid_fn
def insertionSort(numlist, mode=1):
    """
        插入排序： 每次将剩余未排序部分的第一个元素依次与前面排序部分进行比较（从后向前），逐步挪动直到找到比自己小(升序)的元素
        注意:
            插入排序有一种优化算法，叫做拆半插入。因为前面是局部排好的序列，因此可以用折半查找的方法将牌插入到正确的位置，而不是从
            后往前一一比对。折半查找只是减少了比较次数，但是元素的移动次数不变，所以时间复杂度仍为 O(n^2) ！
    """
    
    leng = len(numlist)
    for i in range(1, leng):
        curnum, preInd = numlist[i], i-1
        while preInd >=0 and comp_fn(numlist[preInd], curnum, mode=mode):
            numlist[preInd + 1] = numlist[preInd]
            preInd -= 1
        numlist[preInd+1] = curnum

    pass

@valid_fn
def shellSort(numlist, mode=1):
    """
        shell排序（中文名：希尔排序）：
        算法思想：
            shell排序是进化版的插入排序，每次以一定的间隔gap将待排序列切割成若干子序列，每个自序列进行内部插入排序，
            然后减小gap,重复上述步骤，直至gap=1时，进行最后一次插入排序，相当于简单的插入排序，最终得到有序序列。
        动态间隔序列的选取算法:
           希尔排序的核心在于间隔序列的设定。既可以提前设定好间隔序列，也可以动态的定义间隔序列。下面的代码使用的是
           《算法（第4版》的合著者 Robert Sedgewick 提出的动态定义间隔序列的算法。 
        时间复杂度的计算:(这个比较复杂，没找到详细的推导过程)
    """
    leng = len(numlist)
    gap = 1
    while gap < leng // 3:
        gap = gap * 3 + 1
    
    while gap > 0:
        for i in range(gap, leng):
            curnum, preInd = numlist[i], i - gap
            while preInd >= 0 and comp_fn(numlist[preInd], curnum, mode=mode):
                numlist[preInd + gap] = numlist[preInd]
                preInd -= gap
            numlist[preInd + gap] = curnum
        gap //= 3

    pass


def mergeSort(numlist,mode=1):
    """归并排序
        自上而下的递归式归并排序        
    """
    def merge(left, right, mode=1):
        result = []
        i = j = 0
        while(i < len(left) and j < len(right)):
            if comp_fn(right[j],left[i], mode=mode):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result += left[i:] + right[j:]
        return result
    
    leng = len(numlist)
    if leng <= 1:
        return numlist
    mid = leng // 2 
    left = mergeSort(numlist[:mid], mode=mode)
    right = mergeSort(numlist[mid:],mode=mode)
    return merge(left, right,mode=mode)
    pass

"""
    快速排序
       一种分而治之思想在排序算法上的典型应用。本质上来看，快速排序应该算是在冒泡排序基础上的递归分治法。
       它是处理大数据最快的排序算法之一，虽然 Worst Case 的时间复杂度达到了 O(n²)，但是在大多数情况下都
    比平均时间复杂度为 O(n log n) 的排序算法表现要更好，因为 O(n log n) 记号中隐含的常数因子很小，而且快
    速排序的内循环比大多数排序算法都要短小，这意味着它无论是在理论上还是在实际中都要更快，比复杂度稳定等
    于 O(n log n) 的归并排序要小很多。所以，对绝大多数顺序性较弱的随机数列而言，快速排序总是优于归并排序。
       它的主要缺点是非常脆弱，在实现时要非常小心才能避免低劣的性能。
"""

def quickSort_1(numlist):
    """
        简单版本的快排实现：平均空间复杂度为o(nlogn)
    """
    leng = len(numlist)
    if leng <= 1:
        return numlist
    pivot = numlist[0]
    left = [numlist[i] for i in range(1, leng) if numlist[i] < pivot]
    right = [numlist[i] for i in range(1, leng) if numlist[i] >= pivot]
    return quickSort_1(left) + [pivot] + quickSort_1(right)

def quickSort_2(numlist, left, right):
    """
        复杂版本，但是空间复杂度低，o(logn)
        参数：
            numlist: 待排序的数组
            left: 数组下界（不管是递归内层或外层，边界的索引一律使用原始的numlist的索引）
            right: 数组上界（边界索引备注同上）
    """
    # 分区操作
    def partition(nums,left, right):
        pivot = nums[left]
        while left < right:
            while(left < right and nums[right] >= pivot):
                right -= 1
            nums[left] = nums[right]
            while(left < right and nums[left] < pivot):
                left += 1
            nums[right] = nums[left]
        nums[left] = pivot
        return left
    
    if left < right:
        pivotInd = partition(numlist, left, right)
        quickSort_2(numlist, left, pivotInd-1)
        quickSort_2(numlist, pivotInd+1, right)

    return numlist

    pass


"""
    大根堆：每个节点的值都大于或等于其子节点的值，用于升序排列；
    小根堆：每个节点的值都小于或等于其子节点的值，用于降序排列。
"""    
def heapSort(numlist, mode):
    """
        numlist: 待排序的列表
        mode: 1 表示大根堆，对应升序
              0 表示小根堆，对应降序
    """
    def adjustHeap(nums, i, size):
        """
            进行一次堆调整
            参数：
                nums:待排序的堆，列表形式保存
                i: 当前需要判断调整的非叶子节点
                size: 需要调整的堆的部分的长度，这里主要用来判断下标是否越界（将下标限定在待排序部分）
        """
        lchild = 2 * i + 1 
        rchild = 2 * i + 2
        
        largest = i
        if lchild < size and comp_fn(nums[lchild], nums[largest], mode=mode):
            largest = lchild
        if rchild < size and comp_fn(nums[rchild], nums[largest], mode=mode):
            largest = rchild
        if largest != i:
            nums[i], nums[largest] = nums[largest], nums[i] 
            adjustHeap(nums, largest, size)
        
        pass

    def buildHeap(nums, size):
        for i in range(len(nums)//2)[::-1]:
            adjustHeap(nums, i, size)
        pass

    size = len(numlist)
    buildHeap(numlist, size)
    for i in range(size)[::-1]:
        numlist[0],numlist[i] = numlist[i], numlist[0]
        adjustHeap(numlist, 0, i)
    return numlist
    pass


def countingSort(nums):
    """
        计数排序
    """
    buff = [0] * (max(nums) + 1)
    for num in nums:
        buff[num] += 1
    j = 0
    for i in range(len(buff)):
        while buff[i] > 0:
            nums[j] = i
            buff[i] -= 1
            j += 1
    return nums
    pass

def bucketSort(nums, buckets_size=5):
    """
        桶排序：计数排序的升级版本，将待排序列分配的到若干个桶中，每个桶中进行插入排序，最后再合并
        最好情况：数据能均匀的分到每个桶中
        最坏情况：数据都被分到同一个桶中
        buckets_size: 桶的大小（默认为5）
    """
    minN, maxN = min(nums),max(nums)
    buff = []
    buckets_num = (maxN - minN) // buckets_size + 1  # 将数据分组
    for i in range(buckets_num):
        buff.append([])
    for i in nums:
        buff[(i-minN)//buckets_size].append(i)
    nums.clear()
    for bucket in buff:
        insertionSort(bucket)
        nums.extend(bucket)
    return nums
    pass


def radixSort(nums):
    """
        基数排序：基数排序是桶排序的一种推广，它所考虑的待排记录包含不止一个关键字
        基数排序有两种方法：
            MSD(主位优先法)：从高位开始进行排序
            LSD(次位优先法)：从低位开始进行排序

    """
    mod = 10
    div = 1
    bits = len(str(max(nums)))
    buckets = [[] for _ in range(mod)]
    while bits:
        for num in nums:
            buckets[num // div % mod].append(num)
        #nums.clear()
        j = 0
        for bucket in buckets:
            while bucket:
                nums[j] = bucket.pop(0)
                j+=1
        div *= 10
        bits -= 1
    return nums 
    pass

if __name__ == "__main__":
    # ************ 冒泡排序测试 *******************”
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #Bubble_sort(tt,1)
    #print(tt)
    #Bubble_sort(tt,0)
    #print(tt)

    # ************ 选择排序测试 *******************”
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #selectionSort(tt,1)
    #print(tt)
    #selectionSort(tt,0)
    #print(tt)

    # ************ 插入排序测试 *******************”
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #insertionSort(tt,1)
    #print(tt)
    #insertionSort(tt,0)
    #print(tt)

    # ************ shell排序测试 *******************”
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #shellSort(tt,1)
    #print(tt)
    #shellSort(tt,0)
    #print(tt)

    # ************ 归并排序 ************************
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #tt = mergeSort(tt,1)
    #print(tt)
    #tt = mergeSort(tt,0)
    #print(tt)

    # ************ 快速排序 ************************
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #tt = quickSort_1(tt)
    #print(tt)
    #tt = quickSort_2(tt,0,19)
    #print(tt)

    # ************ 堆排序 ************************
    tt = [randint(10,100) for _ in range(20)]
    print(tt)
    tt = heapSort(tt,1)
    print(tt)
    #tt = heapSort(tt,0)
    #print(tt)

    # ************ 计数排序 ************************
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #tt = countingSort(tt)
    #print(tt)
    #tt = countingSort(tt)
    #print(tt)

    # ************ 桶排序 ************************
    #tt = [randint(10,100) for _ in range(20)]
    #print(tt)
    #tt = bucketSort(tt)
    #print(tt)
    #tt = bucketSort(tt)
    #print(tt)

    # ************ 基数排序 ************************
    #tt = [randint(10,1000) for _ in range(20)]
    #print(tt)
    #tt = radixSort(tt)
    #print(tt)
    #tt = radixSort(tt)
    #print(tt)
    pass
