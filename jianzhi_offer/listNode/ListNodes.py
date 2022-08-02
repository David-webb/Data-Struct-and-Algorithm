# -*- coding:utf-8 -*-


# 链表节点数据结构
class ListNode:
    def __init__(self, x):
        self.val = x
        self.Next = None

def makeListNodes(numlist):
    assert numlist, "列表不能为空！"
    head = ListNode(numlist[0])
    head_bak = head
    for i in numlist[1:]:
        head.Next = ListNode(i)
        head = head.Next
    return head_bak

def showListNodes(lhead):
    tmp_l = []
    while lhead:
        tmp_l.append(lhead.val)
        lhead = lhead.Next
    print(tmp_l)
    


"""
***************************    从尾到头打印链表每个节点的值 ********************************
"""
"""
    方法一： 顺序遍历，list存储，最后反向输出
"""

def tail2front(listNode):
    tmplist = []
    while listNode:
        tmplist.append(listNode.val)
        listNode = listNode.Next
    return tmplist[::-1]


"""
    方法二：顺序遍历，队列左侧依次插入，最后右侧依次出队
"""
from collections import deque
def tail2front_deque(listNode):
    if listNode is None:
        return []
    dq = deque()
    while listNode:
        dq.appendleft(listNode.val)
        listNode = listNode.Next
    return dq





"""
*************************** 输入一个链表，输出该链表中倒数第k个结点。***********************
"""


"""
    方法一：统计链表总长，计算从前向后需要遍历几次取到Kth-tail
"""
def getKthtailnode(head, k):
    c = 0
    head_bak = head
    while head:
        c += 1
        head = head.Next

    c -= k  # 注意这里c不需要加1
    head = head_bak
    if c < 0 or k < 0:
        return None
    
    for _ in range(c): # c条边，刚好能到达kth-tail-node
        head = head.next
    return head

    pass


"""
    方法二：使用快慢指针确定一个长度为K的窗口，快指针在前，慢指针在后，同步往tail走，快指针到tail, 慢指针所指即目标
"""
def getKthtailnode_slide_window(head, k):
    quick_pointer = head
    slow_pointer = head
    k_th = k

    if k < 0:
        return None

    while k > 0 and quick_pointer:
        k -= 1
        quick_pointer = quick_pointer.Next

    if k > 0:   # 链表长度小于K
        return None
    
    while quick_pointer:
        slow_pointer = slow_pointer.Next
        quick_pointer = quick_pointer.Next

    return slow_pointer

    pass




"""
*************************** 输入一个链表，反转链表后，输出链表的所有元素 ***********************
"""

def reverseListNode(head):
    if head is None:
        return None
    tail = head.Next
    head.Next = None
    while tail:
        tmpNode = tail.Next
        tail.Next = head
        head = tail 
        tail = tmpNode
    return head
    pass


"""
*************************************** 合并两个有序链表 ***************************************
    输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
"""
def Merge(p1_Head, p2_Head):
    p1_tmp = p1_Head
    p2_tmp = p2_Head
    if p1_Head.val < p2_Head.val:
        p3 = p1_Head
        p1_tmp = p1_tmp.Next
    else:
        p3 = p2_Head
        p2_tmp = p2_tmp.Next
    p3_head = p3    # 保存一下head
    
    while p1_tmp and p2_tmp:
        if p1_tmp.val < p2_tmp.val:
            p3.Next = p1_tmp
            p1_tmp = p1_tmp.Next
        else:
            p3.Next = p2_tmp
            p2_tmp = p2_tmp.Next
        p3 = p3.Next
    
    if p1_tmp:
        p3.Next = p1_tmp
    
    if p2_tmp:
        p3.Next = p2_tmp

    return p3_head
    


"""
*************************************** 复杂链表的复制 ***************************************
    要求：输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
    返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
    思路：
        第一步：在原始链表的每个节点后面复制一个一样的节点，得到新链表（奇数位是原始节点，偶数位是复制的节点）
        第二步：将偶数位置的新节点的random指针指向random.Next
        第三部：将新链表拆分成原始链表和复制链表
        
    参考：
        https://blog.csdn.net/fuxuemingzhu/article/details/79622359
"""
class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.Next = None
        self.random = None

def Clone(rHead):
    if not rHead:
        return rHead

    rd_node = RandomListNode(rHead.label)
    rd_node.random = rHead.random
    rd_node.Next = Clone(rHead.Next)
    return rd_node
    pass

# 第一步
def cloneNextNodes(rHead):
    if not rHead:
        return rHead
    
    new_head = rHead
    while rHead:
        rd_node = RandomListNode(rHead.label)
        rd_node.random = rHead.random
        rd_node.Next = rHead.Next
        r_Head.Next=rd_node
        r_Head = rd_node.Next

    return new_head

# 第二步
def clonerandomNodes(rHead):
    new_head = rHead
    even_node= rHead.Next
    while even_node:
        even_node.ramdom = even_node.random.Next
        even_node = even_node.Next.Next
    return new_head
    pass

# 第三步
def splitListNodes(rHead):
    ori_head = rHead
    new_head = rHead.Next
    ori_t = ori_head
    new_t = new_head

    while new_t.Next:
        ori_t.Next = new_t.Next
        ori_t = ori_t.Next
        new_t.Next = ori_t.Next
        new_t = new_t.Next

    return ori_head, new_head
    pass



"""
*************************************** 两个链表的第一个公共节点  ***************************************
    两个链表从第一个公共节点开始，往后都是公共节点
    思路：
        方法一：将两个列表节点依次放入stack中，然后从后往前比较，找到第一个不相同的节点，它的next就是第一个公共节点
        方法二（最优解，O(m+n)）：先比较两个链表长度，找出长链表，然后对齐两个链表的尾巴（长链表的工作指针先往后移动多出来的若干个节点），再从左往右依次对比两个链表的节点，直到找到第一个公共节点
"""

def getFirstCommonNode(p1_Head, p2_Head):
    if not p1_Head or not p2_Head:
        return None

    p1 = p1_Head
    p2 = p2_Head
    
    c1 = 0
    while p1:
        c1 += 1
        p1 = p1.Next
    
    c2 = 0
    while p2:
        c2 += 1
        p2 = p2.Next
    
    long_list = p1_Head if c1 > c2 else p2_Head
    short_list = p2_Head if c1 <= c2 else p1_Head

    pl = long_list
    for _ in range(abs(c1 - c2)):
        pl = pl.Next

    sl = short_list
    while pl and sl:
        if pl == sl:
            return pl 
        pl = pl.Next
        sl = sl.Next

    return None

    pass


"""
*************************************** 链表中环的入口结点 ***************************************
    题目：(目前最难的一题)一个链表中包含环，请找出该链表的环的入口结点。
    提示：若单链表有环一定是在尾巴处（准确的说，没有尾巴了）
    思路：快慢指针，快指针步长2，慢指针1，如果遍历到None表示无环，如果相遇表示有环。从相遇的点开始，快指针回到head,慢指针保持在相遇点，二者步长都调整为1，前进知道相遇，相遇的点即入口
    参考：
        https://zhuanlan.zhihu.com/p/33663488
    延伸：
        判断单链表是否有环: 快慢指针（速度比：2:1,如果无环，fast会遍历到None, 如果有环，fast和low会相交 ）
        证明：快慢指针在有环链表中一定会相遇
        证明：求环的入口中的公式
"""
def EntryNodeofLoop(rHead):
    if not rHead or not rHead.Next or rHead.Next.Next:
        return None
    fast = rHead.Next.Next
    slow = rHead.Next
    while(fast != slow):
        if not fast.Next or not fast.Next.Next:
            return None
        fast = fast.Next.Next
        slow = slow.Next

    fast = rHead
    while(fast != slow):
        fast = fast.Next
        slow = slow.Next
    return fast
    pass



"""
*************************************** 删除链表中重复的节点  ***************************************
    在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
"""
def deleteDuplication(rHead):
    first = ListNode(-1) # 定义头指针
    first.Next = rHead
    last = first
    tmp = rHead
    while tmp and tmp.Next:
        if tmp.val == tmp.Next.val:
            test = tmp.Next
            while test and tmp.val == test.val:
                test = test.Next
            tmp.Next = test
        else:
            last.Next = tmp
            last = last.Next
        tmp = tmp.Next
    last.Next = tmp  # 这一步必须要有！！！考虑1，2，3，3，4，4，5，5序列，走一遍即可知
    return first.Next
    pass


if __name__ == "__main__":
    #test_l  = [1,2,3,3,4,4,5, 5]
    #lhead = makeListNodes(test_l)
    #lhead = deleteDuplication(lhead)
    #showListNodes(lhead)

    pass
