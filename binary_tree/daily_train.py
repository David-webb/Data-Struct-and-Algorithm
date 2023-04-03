"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-11-22 09:37
 * Filename      : daily_train.py
 * Description   : 
"""
# -*- coding:utf-8 -*-

class node:
    def __init__(self, data):
        self.val = data
        self.left_node = None
        self.right_node = None
        self.next = None
        pass

class tree: # 实际是二叉树：binary_tree
    def __init__(self, root_data=None):
        if root_data == None:
            self.root = node('#')
        else:
            self.root = node(root_data)
        pass

    def add_node(self, data=None, use_null=False):
        if not use_null:
            assert data, "新节点的数值不能为None!"
        new_node = node(data)
        q = [self.root]
        while q:
            rt = q.pop(0)
            if rt.left_node == None:
                rt.left_node = new_node
                return
            elif rt.right_node == None:
                rt.right_node = new_node
                return 
            else:
                q.append(rt.left_node)
                q.append(rt.right_node)
        pass

   
    def preOrderSearch(self, pnode, ans=[]):
        if pnode is None:
            # return None
            return ans
        else:
            ans.append(pnode.val)

        self.preOrderSearch(pnode.left_node, ans)
        self.preOrderSearch(pnode.right_node, ans)
        return ans
        pass


    def InOrderSearch(self, pnode):
        if pnode:
            self.InOrderSearch(pnode.left_node)
            print(pnode.val)
            self.InOrderSearch(pnode.right_node)
        pass


    def PostOrderSearch(self):
        if pnode:
            self.InOrderSearch(pnode.left_node)
            self.InOrderSearch(pnode.right_node)
            print(pnode.val)
        pass
             

def kill_none_node(root):
    if not root:
        return None
    buffer = [root]
    filter_list = ["#", None]
    while buffer:
        rt = buffer.pop(0)
        if rt.left_node:
            if rt.left_node.val in filter_list:
                rt.left_node=None
            else:
                buffer.append(rt.left_node)

        if rt.right_node:
            if rt.right_node.val in filter_list:
                rt.right_node=None
            else:
                buffer.append(rt.right_node)

    return root


def build_tree(is_full_binaryTree=True):
    # ========== 案例1 ==============
    # val = [1,2,5,3,4,None,6]
    # t = tree(1)
    # for i in range(1,5):
        # t.add_node(val[i])
    # t.add_node(use_null=True)
    # t.add_node(6)

    # ========== 案例2:  ==============
    # val = [1,2,3,4,None,2,4,None,None,None,None,4]
    val = [1,2,5,3,4,None,6]
    rt_v = val.pop(0)
    t = tree(rt_v)
    for i in range(len(val)):
        if val[i]:
            t.add_node(val[i])
        else:
            t.add_node(use_null=True)
    if not is_full_binaryTree:
        kill_none_node(t.root)
    return t
    pass



def maxdepth(node, lastdepth=0):
    """二叉树的最大深度：所谓最大深度就是根节点到「最远」叶子节点的最长路径上的节点数
    """
    if node:
        lastdepth += 1
        lf_d = maxdepth(node.left_node, lastdepth)
        rt_d = maxdepth(node.right_node, lastdepth)
        return lf_d if lf_d > rt_d  else rt_d
    return lastdepth
    pass



def tree_diameter_simplify(root):
    """(优化版)二叉树最长直径：所谓二叉树的「直径」长度，就是任意两个结点之间的路径长度。最长「直径」并不一定要穿过根结点
    思路：左侧的最大深度＋右侧的最大深度
    修正：需要递归（最大直径可能藏在子树中）；后序遍历； 
    改进：自底向上的计算直径，将子树的最大直径缓存下来，这样对于每个父节点来说，只要计算左右两侧的深度叠加，再与历史最大直径比较即可。
    """
    if root:
        lf_d = maxdepth(root.left_node, lastdepth=0)
        rg_d = maxdepth(root.right_node, lastdepth=0)
        return lf_d + rg_d + 1
    else:
        return 0
    pass

def printNodeLayer(root, layer=0):
    """打印每个节点所在层数
    思路：前序遍历
    """
    if root is None:
        return None
    layer += 1
    print(layer)
    printNodeLayer(root.left_child, layer)
    printNodeLayer(root.right_child, layer)
    pass

def printChildrenum(root):
    """打印每个节点的子孙个数
    思路：后续遍历
    """
    if root is None:
        return 0
    left_childs = printChildrenum(root.left_child)
    right_childs = printChildrenum(root.right_child)
    res = left_childs + right_childs
    print(res)
    return res + 1

max_diameter = 0
def treediameter(root):
    """二叉树的最长直径
    """
    maxDepth(root)
    global max_diameter
    print(max_diameter)
    pass

def maxDepth(root):
    if root is None:
        return 0

    left_d = maxDepth(root.left_child)
    right_d = maxDepth(root.right_child)
    res = max(left_d, right_d)
    global max_diameter
    max_diameter = max(left_d + right_d, max_diameter)
    return res + 1
    pass

def findNext(root):
    """填充右侧节点指针
    思路：层序遍历，按照层号对node进行分批，再建立连接
    """
    buf = [root]
    layer_buf = [1]
    layer_d = {}
    while(buf):
        rt = buf.pop(0)
        layer = layer_buf.pop(0) 
        if rt:
            if layer not in layer_d.keys():
                layer_d[layer] = [rt]
            else:
                last = layer_d[layer].pop(-1) # 节省缓存空间
                last.next = rt
                layer_d[layer].append(rt)
            
            buf.extend([rt.left_child, rt.rigth_child])
            layer_buf.extend([layer+1, layer+1])
    
    return root
    pass

# ============================= 2022/12/7 ======================================
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


def frogStep(steps):
    """青蛙跳问题：一只青蛙一次可以跳1阶或2阶台阶，问跳上10阶台阶有几种跳法
    思路(和coins类似)：
        状态是跳的阶数，
        定义dp[steps]为跳上steps台阶的总共的跳法, 则：
            dp[steps] = dp[steps - 1] + dp[steps - 2]
        边界：dp[0] = 0， dp[1] = 1 dp[2] = 2
    """
    dp = [0 ,1, 2]
    if steps <= 2:
        return dp[steps]
    else:
        dp = dp + [0] * (steps-2)
        c = 3
        while c <= steps:
            dp[c] = dp[c-1] + dp[c-2]
            c+= 1
        print(dp)
        return dp[steps]
    pass


def longestIncSubseq(nums):
    """给定一个长度为N的整数数组，求其中最长的递增子序列的长度
    注意：这里的的递增子序列不需要是连续的
    思路：判断是否是动态规划问题：
        穷举：可以。以每个元素为起点遍历整个数组。
        重叠子问题：定义dp[i] = n, 表示以nums[i]为起点的最大递增序列长度为n.
            dp[i] = max(dp[j]) + 1 if num[j] >= num[i] else 0, 其中j in [i+1, n] 
        子问题相互独立：满足
        边界：dp[n] = 1
    params:
        nums是整数数组
    """
    l = len(nums)
    dp = [0] * l 
    dp[-1] = 1
    for i in range(l)[-2::-1]:
        prop_j = [dp[j] for j in range(i+1, l) if nums[j]>=nums[i]]
        if prop_j:
            dp[i] = max(prop_j) + 1 
        else:
            dp[i] = 0
    print(dp, max(dp))
    return max(dp) 
    pass


def crossBridge(T):
    """小朋友过桥:
    一个月黑风高的夜晚，有N个小朋友要过桥，每个人通过的时间不一样，表T记录了所有人过桥的时间，T[i]表示第i个人的过桥时间。桥上一次只能通过不超过两个人，且大家只有一个手电筒，每次通过后需要有人将手电带回来，通过的时间按照两人中最长的算。问所有人最短的通过时间需要多久？
    注意：每次通过后，送手电回来的可以是先前过桥的人！
    初步方案：对T进行排序，耗时最短的两人为s1,s2，那么一轮过桥如下定义：这二人先过桥，s1返回，接着剩下的人中耗时最长的两人过桥，s2返回. 耗时：t = s2 + s1 + max(A,B) + s2 = s2 + s1 + B + s2 (假设B>A)
    从上面看出，最后一轮一定是s1 s2一起过桥。问题在倒数第二轮：
       如果剩下单数人（s1 s2 和一个A）， s1 + A 先过，s1返回；耗时：A + s1
       如果剩下偶数人：和标准程序一样，耗时：t

    在保证s1s2存在的基础上，添加A<B<C过桥，两种情况：
    1. (A,B),C   耗时: t1 = 2*s2 + s1 + B + C + s1
    2. A,(B,C)   耗时: t2 = A + s1 + 2*s2 + s1 + C, 
    t2 - t1 = A - B < 0, 所以，最节省的肯定是A, (B, C)

    综上，
        cost[0] = 0 ; cost[1] = s1 ; cost[2] = s2
        cost[m] = cost[m-1] + T[m-1] + s1   m = 3
        cost[m] = cost[m-2] + T[m] + 2*s2 + s1  m > 3
        即：
        cost[m] = min(cost[m-1] + T[m-1] + s1, cost[m-2] + T[m] + 2*s2 + s1)

    是否是动态规划：
        穷举：满足，可以穷举所有组合情况
        重叠子问题：cost[m] 表示m个人过桥的最短时间, 这里有个前提，对T进行升序排序，然后从前往后加人的
            cost[m] = cost[m-1] + (s1 + T[m]) if m为偶数且m=2 
            cost[m] = cost[m-2] + 2*s2 + T[m] + s1 if m为奇数 
        子问题相互独立：没毛病
        边界：cost[0] = 0 ; cost[1] = s1 ; cost[2] = s2
    """

    T.sort()
    l = len(T)
    cost = [0] * (l+1)
    cost[1] = T[0]
    cost[2] = T[1]
    
    for i in range(3,l+1):
        if i == 3: # m+1为奇数，m为偶数
            cost[i] = cost[i-1] + T[0] + T[i-1] # 这里使用的是T[i-1],因为cost和T的索引上错位gap是1，cost比T多一个元素
        else:
            cost[i] = cost[i-2] + 2*T[1] + T[i-1] + T[0] 
    print(cost) 
    return cost[-1]

def cutline(n):
    """切割钢条，使得利润最大，给出切割方案
       定义： profit[n] 表示长度为n钢条最佳利润
       profit[n] = max(profit[n-i] + cost[i])   0<i<=n
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


def flipBT(root):
    """翻转二叉树
    """
    if root is None:
        return

    l = root.left_node
    root.left_node = root.right_node
    root.right_node = l

    flipBT(root.left_node)
    flipBT(root.right_node)


def left2right_TS(root):
    if root is None:
        return None
    assert root.right_node is None, "found some node has right child, sth went wrong with right2left_TS! and the node val is " + str(root.val) 
    root.right_node = root.left_node
    root.left_node = None
    left2right_TS(root.right_node)
    return root
    pass

def treeSqueeze(tree_handler):
    """二叉树转链表
    思路：
        先序遍历
        将每个节点的右孩子链接到左孩子的先序末端位置（此时所有节点都没有右孩子）
        将所有节点的左孩子都挪到右孩子的位置
    """
    t, root = tree_handler, tree_handler.root
    right2left_TS(root)
    printTree(t, 'full')
    left2right_TS(root) 
    # return root
    printTree(t, 'right_slide')
    pass


def right2left_TS(root):
    if root is None:
        return None
    if root.left_node:
        left_tail = right2left_TS(root.left_node)
        # tail.left_node = root.right_node
        # root.right_node = None
    else:
        # root.left_node = root.right_node
        # root.right_node = None
        left_tail = root

    if root.right_node:
        right_tail = right2left_TS(root.right_node)
    else:
        right_tail = None

    left_tail.left_node = root.right_node
    root.right_node = None

    # if left_tail.left_node:
        # return left_tail.left_node
    # else:
        # return left_tail 

    if right_tail:
        return right_tail
    else: # right_tail 为None只有当right_node为None的时候才会出现
        return left_tail
    pass


def BT2line(root):
    """二叉树转链表：
    按照先序遍历顺序将二叉树转成右侧单链表（要求使用inplace操作）
    思路：因为是先序遍历，所以右孩子在左孩子之后，所以操作步骤如下：
        1. 将所有节点的右孩子转到左孩子的末尾, 保证先序顺序（此时右孩子都为空）
        2. 将所有的左孩子转移到右孩子位置
    """
    right2left(root) # 保证链表顺序
    # printTree(root)
    reshape(root) # 撸成右侧单链表
    printTree(root)
    pass

def right2left(root):

    if root is None:
        return None

    l_tail_node = right2left(root.left_node)
    r_tail_node = right2left(root.right_node)

    if l_tail_node:
        l_tail_node.left_node = root.right_node
        root.right_node = None
    elif r_tail_node:
        root.left_node = root.right_node
        root.right_node = None

    return r_tail_node if r_tail_node else root
    pass

def reshape(root):
    """将所有节点的孩子节点的左孩子都换到右孩子节点
        注意，要先验证右孩子部分都是空
    """
    if root is None:
        return 

    assert root.right_node is None, "sth wrong with right2left! " 
    root.right_node = root.left_node
    root.left_node = None 
    reshape(root.right_node)
   
def printTree(tree_h, mode='full'):
    # buf = [root]
    # ans = []
    # while buf:
        # rt =buf.pop(0)
        # if rt is None or rt.val is None:
            # ans.append(None)
        # else:
            # ans.append(rt.val)
            # buf.append(rt.left_node)
            # buf.append(rt.right_node)

        # # if rt.left_node: # is None or rt.left_node.val is None:
            # # buf.append(rt.left_node)
        # # if rt.right_node: #  is None or rt.right_node.val is None:
            # # buf.append(rt.right_node)
    # print(ans)
    if isinstance(tree_h, tree):
        root = tree_h.root
    else:
        root = tree_h

    if mode == "right_slide":
        if root:
            print(root.val)
            printTree(root.right_node,  mode=mode)
    elif mode == "full":
        ans = tree_h.preOrderSearch(root)
        print(ans)
        pass


def one_zero_bag():
    """0-1背包
    
    """
    pass

def merge_sort_0403(nums):

    pass

if __name__ == "__main__":
    # ========= 测试 go_linkedsqueezing函数 ============
    # 构建树（注意节点的val数值要和题目一致!）
    # Tree = build_tree()
    # root = go_linkedsqueezing(Tree.root)
    # print(leveltravers(Tree.root))
    # print(Tree.preOrderSearch(Tree.root))

    # ============测试：tree_Duplicated_subtrees ============
    # Tree = build_tree(is_full_binaryTree=False)
    # 层序遍历的测试
    # print(leveltravers(Tree.root))
    # 最大重复子树的测试
    # ans = tree_Duplicated_subtrees(Tree.root)
    # print(list(ans))
    # 二叉树直径测试
    # print(tree_diameter_simplify(Tree.root))

    # ============= 测试：coins ====================
    # coins(11)
    # ============= 测试：frogSteps ====================
    # frogStep(10)
    # ============= 测试：longestIncSubseq ====================
    # nums = [10,9,2,5,3,7,101,18]
    # longestIncSubseq(nums)
    # ============= 测试：crossBridge ====================
    # L = [3,1,2,5,10,11, 34, 56]
    # crossBridge(L)
    # ============= 测试：cutline ====================
    # cutline(10)
    # ============= 测试：BT2line ====================
    # 测试列表：[1,2,5,3,4,None,6]
    # t = build_tree(False)
    # BT2line(t.root)
    # 另一次训练尝试
    # t = build_tree(False)
    # treeSqueeze(t)
    # ============= 测试：
    pass



    


