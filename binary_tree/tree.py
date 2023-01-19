"""
 * Author        : TengWei
 * Email         : davidwei.teng@foxmail.com
 * Created time  : 2022-05-24 08:32
 * Filename      : tree.py
 * Description   : 
"""
# -*- coding:utf-8 -*-

import collections

class node():
    def __init__(self, data="#"):
        self.val = data if data else "#"
        self.left_child = None
        self.right_child = None
        self.next = None # 右侧指针（完美二叉树的层序遍历中，指向每个节点所在层的右侧节点）
        self.parent = None
        pass

class tree():
    def __init__(self, root_data=None):
        self.root = node(root_data)
        # self.maxtreedepth = 0
        pass

    # 二叉树中添加新的节点(按照BFS的顺序)
    def add_node(self, data=None, use_null=False):
        if not use_null:
            assert data, "新节点的数值不能为None!"
        q = [self.root]
        tnode = node(data)
        while(q):
            t_root = q.pop(0)
            # if t_root.val == "#":
                # self.root = tnode
                # return 
            if t_root.left_child is None:
                t_root.left_child = tnode 
                return
            elif t_root.right_child is None:
                t_root.right_child = tnode 
                return
            else:
                q.append(t_root.left_child)
                q.append(t_root.right_child)
        pass

    def preOrderSearch(self, pnode, ans=[]):
        # if pnode:
            # # print(pnode.val)
            # ans.append(pnode.val)
            # if pnode.left_child:
                # self.preOrderSearch(pnode.left_child, ans)
            # if pnode.right_child:
                # self.preOrderSearch(pnode.right_child, ans)
        # return ans

        # ================ 简化版 ==========================
        if pnode is None:
            # return None
            return ans
        else:
            ans.append(pnode.val)

        self.preOrderSearch(pnode.left_child, ans)
        self.preOrderSearch(pnode.right_child, ans)
        return ans
        pass

    def InOrderSearch(self, pnode):
        # if pnode:
            # if pnode.left_child:
                # self.InOrderSearch(pnode.left_child)

            # print(pnode.val)

            # if pnode.right_child:
                # self.InOrderSearch(pnode.right_child)

        # ================ 简化版 ==========================
        if pnode:
            self.InOrderSearch(pnode.left_child)
            print(pnode.val)
            self.InOrderSearch(pnode.right_child)
        pass

    def PostOrderSearch(self):
        # if pnode:
            # if pnode.left_child:
                # self.PostOrderSearch(pnode.left_child)
            # if pnode.right_child:
                # self.PostOrderSearch(pnode.right_child)
            # print(pnode.val)

        # ================ 简化版 ==========================
        if pnode:
            self.InOrderSearch(pnode.left_child)
            self.InOrderSearch(pnode.right_child)
            print(pnode.val)
        pass


def maxdepth(node, diameter=False):
    """二叉树的最大深度：所谓最大深度就是根节点到「最远」叶子节点的最长路径上的节点数
    """
    # depth = lastdepth + 1 if node else lastdepth
    # lnode, rnode = node.left, node.right
    # # while(lnode or rnode):
        # # depth += 1
        # # lnode 
    # ldp = rdp = -1
    # if lnode:
        # ldp = maxdepth(lnode, depth)
    # if rnode:
        # rdp = maxdepth(rnode, depth)
    # return max(ldp, rdp, depth)

    # ================ 简化版 ==========================

    if node is None:
        return 0
    lf_d = maxdepth(node.left_child, diameter=diameter)
    rt_d = maxdepth(node.right_child, diameter=diameter)
    if diameter:
        my_diameter = lf_d + rt_d
        global max_diameter
        max_diameter = max(max_diameter, my_diameter)
    return 1 + max(lf_d, rt_d)

def tree_diameter(root):
    """二叉树最长直径：所谓二叉树的「直径」长度，就是任意两个结点之间的路径长度。最长「直径」并不一定要穿过根结点
    思路：max(左侧的最大深度＋右侧的最大深度, 左侧的最大直径，右侧的最大直径）
    修正：需要递归（最大直径可能藏在子树中）；后序遍历； 
    """
    if not root:
        return 0
    lnode, rnode = root.left, root.right
    lmaxdep = rmaxdep = 0
    l_dia = r_dia = 0
    if lnode:
        lmaxdep = maxdepth(0, lnode)
        lmaxdep = max(lmaxdep, 1) # 1是因为root到lnode的长度为1
        l_dia = tree_diameter(lnode)
    if rnode:
        rmaxdep = maxdepth(0, rnode)
        rmaxdep = max(rmaxdep, 1) # 1是因为root到rnode的长度为1
        r_dia = tree_diameter(rnode)

    ans = max(lmaxdep + rmaxdep, l_dia, r_dia)
    return ans 
    pass

max_diameter = 0
def tree_diameter_simplify(root):
    """(优化版)二叉树最长直径：所谓二叉树的「直径」长度，就是任意两个结点之间的路径长度。最长「直径」并不一定要穿过根结点
    思路：左侧的最大深度＋右侧的最大深度
    修正：需要递归（最大直径可能藏在子树中）；后序遍历； 
    改进：自底向上的计算直径，将子树的最大直径缓存下来，这样对于每个父节点来说，只要计算左右两侧的深度叠加，再与历史最大直径比较即可。
    """
    max_depth(root, diameter=True)
    global max_diameter
    return max_diameter
    
def leveltravers(root):
    """层序遍历(BFS)"""
    ans = [root.val]
    buffer = [root]
    while buffer:
        rt = buffer.pop(0) 
        if rt.left_child:
            ans.append(rt.left_child.val)
            buffer.append(rt.left_child)
        if rt.right_child:
            ans.append(rt.right_child.val)
            buffer.append(rt.right_child)
    return ans
    pass


resval = []
def leveltravers_v2(curnodelist):
    """层序遍历(BFS)的第二种解法
    相当于按层保存了node的数值
    params:
        curnodelist: list(node)
    """
    nodevalist = []
    nextnodelist = []
    for node in curnodelist:
        if node: # 这里默认node是可以为None的，这样可以构建完美二叉树
            # print(node.val)
            nodevalist.append(node.val)
        else:
            nodevalist.append('#') # node缺失标记
        # if node.left:
            # nextnodelist.append(node.left)
        # if node.right:
           # nextnodelist.append(node.right)
        nextnodelist.append(node.left)
        nextnodelist.append(node.right)
    global resval
    resval.append(nodevalist)
    leveltravers_v2(nextnodelist)
    pass

def flip_tree(root):
    """翻转二叉树
    思路：递归的将每个节点的左右子节点互换
    """ 
    buffer = [root]
    while buffer:
        rt = buffer.pop(0)
        tmp = rt.left_child # 即使是None也要转换
        rt.left_child = rt.right_child
        rt.right_child = tmp
        if rt.left_child:
            buffer.append(rt.left_child)
        if rt.right_child:
            buffer.append(rt.right_child)
    pass

def go_linkedsqueezing(root):
    """按照前序遍历的顺序将二叉树转为(右)链表(inplace)
    思路：分解问题，自底向上，将每个node的左分支变成右链表，将右分支变成右链表，再将右节点接入左分支链表的尾部，再将左节点换到右节点即可
    """
    tree_squeeze2linked(root)
    left2right(root)
    return root
    

def left2right(root):
    """自定向下：将每个节点的左孩子统一转到右孩子位置"""
    if not root:
        return None
    buffer = [root]
    while buffer:
        rt = buffer.pop(0)
        if rt.left_child:
            assert not rt.right_child, "tree_squeeze2linked 函数存在缺陷！"
            rt.right_child = rt.left_child 
            rt.left_child = None
            buffer.append(rt.right_child)
        elif rt.right_child:
            assert not rt.left_child, "tree_squeeze2linked 函数存在缺陷！"
            buffer.append(rt.right_child)
        # else:
    pass

def tree_squeeze2linked(root):
    """将树转化成链表（限制为in-place操作）
    自底向上：将每个node的右分支接到左分支的尾部的有孩子节点
    """
    # ans_node = None
    if root:
        left_tail = right_tail = None
        if root.left_child:
            left_tail = tree_squeeze2linked(root.left_child)
        # print(root.val)

        if root.right_child:
            right_tail = tree_squeeze2linked(root.right_child)
            # if left_tail:
                # left_tail.right_child = root.right_child
                # root.right_child = None
        if left_tail:
            left_tail.right_child = root.right_child
            root.right_child = None

        if right_tail:
            return right_tail 
        elif left_tail:
            return left_tail
        # else:
            # return root
    return root

def kill_none_node(root):
    if not root:
        return None
    buffer = [root]
    while buffer:
        rt = buffer.pop(0)
        if rt.left_child:
            if rt.left_child.val == "#":
                rt.left_child=None
            else:
                buffer.append(rt.left_child)

        if rt.right_child:
            if rt.right_child.val == "#":
                rt.right_child=None
            else:
                buffer.append(rt.right_child)

    return root

    pass

def build_tree(is_full_binaryTree=True):
    # ========== 案例1 ==============
    # val = [1,2,5,3,4,None,6]
    # t = tree(1)
    # for i in range(1,5):
        # t.add_node(val[i])
    # t.add_node(use_null=True)
    # t.add_node(6)

    # ========== 案例2:  ==============
    val = [1,2,3,4,None,2,4,None,None,None,None,4]
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


def fill_next(root):
    """填充节点的右侧指针
    """
    ans = [root.val]
    buffer = [root]
    level = [0]
    while buffer:
        rt = buffer.pop(0) 
        l = level.pop(0)
        if rt.left_child:
            ans.append(rt.left_child.val)
            buffer.append(rt.left_child)
            level.append(l+1)
        if rt.right_child:
            ans.append(rt.right_child.val)
            buffer.append(rt.right_child)
            level.append(l+1)
        if buffer[0] and level[0] == l:
            rt.next = buffer[0] # 否则的话，rt.next不需要赋值，默认为None
    pass

def constructMaximumBinaryTree(numlist):
    """最大二叉树
    规则：根据numlist中的最大值作为root，最大值左边的数值构建最大二叉子树作为左孩子，右边构建最大二叉子树作为右孩子
    Args:
        存放不重复数值的list
    """
    if not numlist:
        return None

    rt = max(numlist)
    rt_idx = numlist.index(rt)
    # t = tree(rt) 
    # root = t.root 
    root = node(rt)
    l = numlist[:rt_idx]
    r = numlist[rt_idx+1:] if rt_idx < (len(numlist)-1) else []
    root.left_child = constructMaximumBinaryTree(l)
    root.right_child = constructMaximumBinaryTree(r)
    return root
    pass

def build_tree_with_pre_and_in_order(preOrder, inOrder):
    """根据给出的前序遍历和中序遍历序列还原出二叉树结构
    预设前提：树中没有重复的元素
    """
    if not preOrder or not inOrder:
        return None
    v = preOrder.pop(0)
    rt = node(v)
    idx = inOrder.index(v)
    in_l = inOrder[:idx]
    in_r = inOrder[idx+1:] if idx+1 < len(inOrder)-1 else []
    pre_l = [i for i in preOrder if i in in_l] # 要保持preOrder中元素顺序
    pre_r = [j for j in preOrder if j in in_r] # 同上
    left_c = build_tree_with_pre_and_in_order(pre_l, in_l)
    right_c = build_tree_with_pre_and_in_order(pre_r, in_r)
    rt.left_child = left_c
    rt.right_child = right_c
    return rt
    pass

def build_tree_with_in_and_post_order(inOrder, postOrder):
    """根据给出的后序遍历和中序遍历序列还原出二叉树结构
    预设前提：树中没有重复的元素
    """
    if not inOrder or not postOrder:
        return None
    v = postOrder.pop(-1)
    rt = node(v)
    idx = inOrder.index(v)
    
    in_l = inOrder[:idx]
    in_r = inOrder[idx+1:] if idx+1 < len(inOrder)-1 else []

    post_l = [i for i in postOrder if i in in_l]
    post_r = [j for j in postOrder if j in in_r]

    left_c = build_tree_with_in_and_post_order(in_l, post_l)
    right_c = build_tree_with_in_and_post_order(in_r, post_r)
    rt.left_child = left_c
    rt.right_child = right_c
    return rt

def build_tree_with_pre_and_post_order(preOrder, postOrder):
    """根据给出的前序遍历和后序遍历序列还原出二叉树结构
    预设前提：
        i. 树中没有重复的元素
        ii.如果结果不唯一，返回其中任意一个
    """
    if not preOrder or not postOrder:
        return None
    
    v = preOrder.pop(0)
    rt = node(v)
    idx = postOrder.index(v)
    child_nodes = postOrder[:idx]
    bro_nodes = postOrder[idx+1:] if idx+1 < len(inOrder)-1 else []

    child_pre = [n for n in preOrder if n in child_nodes]
    bro_pre = [n for n in preOrder if n in bro_nodes]
    # 构建子树
    if len(child_nodes)<=2:
        c = 0
        for i in child_nodes:
            if c == 0:
                rt.left_child = node(i)
            else:
                rt.right_child = node(i)
            c+=1 
    else:
        lchild, rchild = build_tree_with_pre_and_post_order(child_pre, child_nodes)
        rt.left_child = lchild
        rt.right_child = rchild

    # 构建兄弟结构
    if bro_nodes:
        bro_node = build_tree_with_pre_and_post_order(bro_pre, child_nodes)
    else:
        bro_node = None

    return rt, bro_node


def tree_serialize(root):
    """二叉树的序列化
    思路：这里使用层序遍历
    """
    ans = []
    if not root:
        return ['#']

    buffer = [root]
    while buffer:
        rt = buffer.pop(0)
        # 添加左子树
        if rt.left_child:
            buffer.append(rt.left_child)
            ans.append(rt.left_child.val)
        else:
            ans.append("#")

        # 添加右子树
        if rt.right_child:
            buffer.append(rt.right_child)
            ans.append(rt.right_child.val)
        else:
            ans.append("#")
    return ans
    pass

def tree_deserialize(data):
    """二叉树的反序列化
    思路：这里遵循序列化的选择，默认是层序遍历顺序
    """
    if not data:
        return [] 
    buffer = []
    rt_s = node(data.pop(0))
    buffer.append(rt_s)
    while(buffer):
        rt = buffer.pop(0)
        if data:
            l_v = data.pop(0)
            if l_v != "#":
                t_node = node(l_v)
                buffer.append(t_node)
                rt.left_child = t_node

        if data:
            r_v = data.pop(0)
            if r_v != "#":
                t_node = node(r_v)
                buffer.append(t_node)
                rt.right_child = t_node    
    return rt_s

# 本题较难，抄的作业!!!!!!!!!!!!!
def tree_Duplicated_subtrees(root):
    """寻找二叉树中重复出现的子树，返回重复类型中任意一个实例的root即可
    思路：
        对树的所有子树做序列化，然后将序列化的字符串存到一个字典中并记录出现次数，如果出现次数为2，表示出现重复子树
    """
    d = collections.defaultdict(list)
    def dfs(root):
        if root:
            res = (root.val, dfs(root.left_child), dfs(root.right_child))
            d[res].append(root)
            return res
        return None
    dfs(root)
    # return [v[0].val for k,v in d.items() if len(v)>1]
    ans = [v for k,v in d.items() if len(v)>1]
    t_ans = []
    for a in ans:
        tmp = []
        for it in a:
            tmp.append(it.val)
        t_ans.append(tmp)
    return t_ans
    pass

# ==================== 归并排序的应用1：计算一维数组中每个元素右侧比它小的元素个数 
class pair:
    def __init__(self, val, idx):
        self.idx = idx
        self.val = val
        pass


def MergeSort(nums, count):
    """归并排序(改版,针对归并排序的应用做了修改)
    params:
        nums：待排序的数组(list)
        count: 个数统计数组
    """
    hi = len(nums)
    # print("in MergeSort:", hi)
    if hi <= 1:
        return nums
    mid = hi // 2
    left_n = MergeSort(nums[:mid], count)
    right_n = MergeSort(nums[mid:], count)
    
    # 合并两个列表: merge(left_n, right_n)
    i = j = 0
    lf = len(left_n)
    rg = len(right_n)
    ans = []
    # print(lf, rg)
    while(i < lf and j < rg):
        if left_n[i].val > right_n[j].val:
            ans.append(right_n[j])
            j += 1
        else:
            ans.append(left_n[i])
            count[left_n[i].idx] += j
            i += 1
    # print(" ".join([str(t) for t in count]))

    if i < lf: # 右侧序列已经全部归并完毕,左侧还有空
        ans = ans + left_n[i:]
        for t in left_n[i:]:
            count[t.idx] += j # 此时的j应该是len(right_n), 所以不需要用j+1
    if j < rg: # left_n已经归并完毕，但右侧没有，此时不要统计右侧内部的元素计数（右侧模块形成前已经统计过了）
        ans = ans + right_n[j:]
    
    # print(" ".join([str(t) for t in count]))
    return ans
    pass

def mergeCount(nums):
    """计算数组中每个元素右侧小于当前元素的个数
    思路：归并排序使用的递归merge来实现，每次merge的左右两块内部都是升序的，
    因此，对于左侧块内部的元素来说，右侧比它小的元素隐藏在右侧块中，右侧块内部的元素在右侧块形成之前的merge中进行统计
    """
    hi = len(nums)
    assert hi>0, "数组不能为空！"
    if hi == 1:
        return [0]
    nums_p = [pair(t,i) for i, t in enumerate(nums)]
    count = [0] * hi # 预先分配缓存
    MergeSort(nums_p, count)
    print(count)
    pass

# ==================== 归并排序的应用2：翻转对
"""给定一个数组nums, 如果i<j, 且 nums[i] > 2* nums[j],我们就将(i,j)称作一个重要翻转对
你需要返回给定数组中的重要翻转对的数量.
"""
flip_pair_count = 0
def MergeSort_app2(nums):
    hi = len(nums)
    # print("in MergeSort:", hi)
    if hi <= 1:
        return nums
    mid = hi // 2
    left_n = MergeSort_app2(nums[:mid])
    right_n = MergeSort_app2(nums[mid:])
    
    # 高效计算翻转对的个数
    lf = len(left_n)
    rg = len(right_n)

    end = 0
    global flip_pair_count
    for l in range(lf):
        while(end < rg and left_n[l] > right_n[end] * 2): # 这里有可能数值比较大，乘以2后可能会溢出，最好强制转换成long
            end += 1 
        flip_pair_count += end

    # 合并两个列表: merge(left_n, right_n)
    i = j = 0
    ans = []
    # print(lf, rg)
    while(i < lf and j < rg):
        if left_n[i] > right_n[j]:
            ans.append(right_n[j])
            j += 1
        else:
            ans.append(left_n[i])
            i += 1
    # print(" ".join([str(t) for t in count]))

    if i < lf: # 右侧序列已经全部归并完毕,左侧还有空
        ans = ans + left_n[i:]
    if j < rg: # left_n已经归并完毕，但右侧没有，此时不要统计右侧内部的元素计数（右侧模块形成前已经统计过了）
        ans = ans + right_n[j:]
    
    # print(" ".join([str(t) for t in count]))
    return ans
    pass

    pass

# ==================== 归并排序的应用2：区间和的个数
"""给定一个整数数组nums以及两个证书lower和upper. 求数组中，值位于范围[lower, upper]（闭区间）之内的区间和的个数。
区间和S(i,j)表示在nums中，位置从i到j的元素只和，包含i和j(i <= j).
"""
lower = upper = None 
count = 0
def countrangesum(nums, low, upr):
    global lower, upper, count
    lower = low
    upper = upr
    pre_sum = [0] * (len(nums)+1)
    for i in range(len(nums)): # 这里pre_sum多了一个元素，只是为了方便通式的实现，并不影响最终结果（初始值为0）
        pre_sum[i+1] = pre_sum[i] + nums[i]
    mergesort_app3(pre_sum)
    print(count)
    pass

def mergesort_app3(nums):
    """这里有几点思考注意点：归并排序对前缀和序列的右半部分的排序的操作确实会改变右侧的连续性，但本题是直接和左侧元素相减的，不用考虑右侧的连续性
    """
    hi = len(nums)
    # print("in MergeSort:", hi)
    if hi <= 1:
        return nums
    mid = hi // 2
    left_n = mergesort_app3(nums[:mid])
    right_n = mergesort_app3(nums[mid:])
    
    # 高效计算翻转对的个数
    lf = len(left_n)
    rg = len(right_n)
    global lower, upper, count
    # print("start:",count)
    start = end = 0 
    for l in range(lf):
        while(start<rg and (right_n[start] - left_n[l]) < lower):
            start += 1

        while(end<rg and (right_n[end] - left_n[l]) <= upper):
            end += 1
        count += (end - start)

    # print("end:",count)
    # 合并两个列表: merge(left_n, right_n)
    i = j = 0
    ans = []
    # print(lf, rg)
    while(i < lf and j < rg):
        if left_n[i] > right_n[j]:
            ans.append(right_n[j])
            j += 1
        else:
            ans.append(left_n[i])
            i += 1
    # print(" ".join([str(t) for t in count]))

    if i < lf: # 右侧序列已经全部归并完毕,左侧还有空
        ans = ans + left_n[i:]
    if j < rg: # left_n已经归并完毕，但右侧没有，此时不要统计右侧内部的元素计数（右侧模块形成前已经统计过了）
        ans = ans + right_n[j:]
    
    # print(" ".join([str(t) for t in count]))
    return ans
    pass

# ========================================== 二叉搜索树 ==========================================
def kthsmallest(root, k, c=0):
    """二叉搜索树中的第K小的元素
    思路：中序遍历即可
    """
    if root == None:
        return c 
    c = mid_traverse(root.left_child, c, k)
    c += 1
    if c == k:
        return root
    c = kthsmallest(root.right_child, c, k)
    return c
    pass

def kthsmallest_v2(root, k, c=0):
    """二叉搜索树中的第K小的元素
    思路：在构建BST的时候,为每个node额外维护一个变量size，用于记录当前node为根的二叉树的node总个数, 这样在判断kth的时候，直接做比较就行了（类似二分查找），效率是O(logN)
    但是BST的增删操作需要对每个节点数据进行维护,不是太方便，这里不做实现
    """
    pass


tsum = 0
def convertBST(root):
    """(没做出来)将BST转化为累加树(每个节点值更新为原来BST中数值大于等于该节点的数值之和)
    """
    if root is None:
        return
    rg = convertBST(root.right_child)
    tsum += root.val
    root.val = tsum
    lf = convertBST(root.left_child)
    pass
      

def isValidBST(root, min_v, max_v):
    """验证二叉搜索树的合法性
    params:
        root:
        min_v: 当前节点root的左侧分支最小节点数值, root取值下限
        max_v: 当前节点root的右侧分支最大节点数值，root取值上限
    """
    if root == None:
        return True
    if min_v != None and root.val <= min_v:
        return False
    if max_v != None and root.val >= max_v:
        return False
    return isValidBST(root.left, min_v, root) and isValidBST(root.right,root, max_v)
    pass

pre_num = float('-inf')
ans = True
def isValidBST_v2(root):
    """判断BST合法性：使用inorder遍历，保证数据的有序性（递增或递减）
    """
    if root is None:
        return 
    global pre_num, ans 
    isValidBST_v2(root.left)
    if pre_num == float('-inf'):
        pre_num = root.val
    if root.val <= pre_num: # 这里的等于根据实际要求可以选择去除
        ans = False
    else:
        pre_num = root.val
    isValidBST_v2(root.right)

def searchBST(node, target):
    """在BST中搜索值为target的节点
    """
    if node is None:
        return None
    elif target < node.val:
        return searchBST(node.left, target)
    else:
        return searchBST(node.right, target)
    return node
    pass

def insert_into_BST(root, val):
    """（没做出来）在BST中插入一个数
    """
    if root is None:
        return node(val)
    if root.val > val:
        root.left = insert_into_BST(root.left_child, val)
    if root.val < val:
        root.right = insert_into_BST(root.right_child, val)
    return root
    pass

def get_rightchild_min(right_child):
    """找到右侧分支的最左侧节点（用于BST中）"""
    while(right_child.left_child):
        right_child = right_child.left_child
    return right_child
    pass

def delete_node(root, key):
    """(困难)从BST中删除指定数值的节点（BST中通常不存在重复数值的节点）
    思路：分三种情况：
        情况1：该节点是叶子节点: 直接删除即可
        情况2：该节点只有单侧分支: 用直系childnode代替被删除的节点
        情况3：该节点有量侧分支: 有两种方案，要么选择左侧分支的最右侧节点，要么选择右侧分支的最左侧节点代替被删除节点(这里选择后一种方案)
    params:
        root: BST的根节点
        key: 需要删除节点的数值
    思考：为什么不直接通过交换数值val的方式来删除节点
    答：首先，对于情况3，会涉及分支结构的调整，仅仅是节点换位置还不够
    另外，更重要的，在实际应用场景中，val大多是非常复杂的数据结构，直接交换比较复杂。而节点的链式操作可以实现和数据的解耦
    """
    if root is None:
        return None
    if root.val == key:
        # ========== 解决情况1和2 ======================
        if root.left_child is None:
            return root.right_child # 可以是None
        if root.right_child is None:
            return root.left_child # 可以是None

        # ========== 解决情况3 =========================
        proposal_node = get_rightchild_min(root.right_child) # 获取右侧分支数值最小的节点，并缓存为A
        root.right_child = delete_node(root.right_child, proposal_node.val) # 将最小节点从当前节点的右侧分支中删除, 此时A为游离节点
        proposal_node.left_child = root.left_child # 将当前节点的左右分支分别接到缓存节点A上
        proposal_node.right_child = root.right_child
        root = proposal_node # 用缓存节点代替当前被删除节点，返回 (注意，此时还没有构建和父节点的连接)
    elif root.val > key: # 要删除的点在左侧分支
        root.left_child = delete_node(root.left_child, key)   # 被删除的节点及其分支重构后，在这里和父节点建立连接
    elif root.val < key: # 要删除的点在右侧分支
        root.right_child = delete_node(root.right_child, key) 
    return root
    pass

# ======================== BST的构建 ============================
import numpy as np
memo = None
def num_of_BST_trees(n):
    """(难，没写出来)给定整数数值n，问以(1,2,...,n)构建BST,不同的类型个数
    思路：这是个动态规划的问题
    """
    global memo
    memo = np.zeros((n,n))
    return count_BST_nums(1, n)
    pass

def count_BST_nums(start, end):
    s, e = start, end
    if s > e:
        return 1 # 这种情况对应BST是空

    if memo[start, end]:
        return memo[start, end]

    sum_bst = 0
    for i in range(s, e+1):
        left_c = count_BST_nums(s, i-1)
        right_c = count_BST_nums(i+1, e)
        sum_bst += left_c * right_c
    memo[s,e] = sum_bst
    return sum_bst
    pass

def generate_BST(n):
    """(难，没有做出来) 给定整数数值n，以(1,2,...,n)构建BST生成所有的BST
    思路：
    """
    if n == 0:
        return []

    return build_bst(1, n)

def build_bst(s, e):
    res = []
    if s < e:
        res.append(null)
        return res

    for i in range(s, e+1):
        left_t = build_bst(1, i-1)
        right_t = build_bst(i+1, e)
        for lt in left_t:
            for rt in right_t:
                new_root = node(i)
                new_root.left_child = lt
                new_root.right_child = rt
                res.append(new_root)

    return res

# ======================== 计算完全二叉树的节点个数 =========================
import math
def count_fullbinarytree_nodes(root):
    """计算完全二叉树的节点个数，
    思路：可以遍历一遍,O(N)；也可以有更高效的方法，O(logN * logN),思路如下：
        首先，区分一下满二叉树和完全二叉树，前者是一种特殊的完全二叉树，每一层都是满的。如果是满二叉树，其节点个数可以直接计算：2^h - 1, 其中h是满二叉树的高度. 
        其次，一个完全二叉树的两个子树，至少有一个是满二叉树
    """
    l = r = root
    hl = hr = 0
    while(l):
        l = l.left_child
        hl += 1
    
    while(r):
        r = r.left_child
        hr += 1

    if (hl == hr):
        return math.pow(2, hl)  - 1

    # 如果左右分支高度不同，按照普通二叉树的逻辑计算
    return 1 + count_fullbinarytree_nodes(root.left_child) + count_fullbinarytree_nodes(root.right_child)
    pass


# ======================== 最邻近公共祖先（LCA: Lowest Common Ancestor） =========================
def lowestCommonAncestor(root, p, q):
    """给一棵元素各不相同的二叉树，求解其中数值为p和q的两个节点的最近公共祖先节点
    思路：
    情况1：如果一个节点的左右子树分别包含这两个元素，那么该节点就是p和q的LCA
    情况2：LCA就是p或q, 另一个节点是LCA的子孙节点
    """
    if root is None:
        return None

    if root.val in [p, q]: # 需要保证所有的目标节点都存在
        return root

    left_ = lowestCommonAncestor(root.left_child, p, q)
    right_ = lowestCommonAncestor(root.right_child, p, q)
    
    if left_ and right_:
        return root

    return left_ if right_ is None else right_
    pass

def lowestCommonAncestor_v2(root, val_list):
    """给一棵元素各不相同的二叉树，求解其中数值为val_list中所有节点的最近公共祖先节点(val列表都有对应的节点)
    思路：
        情况1：如果一个节点的左右子树分别包含这两个元素，那么该节点就是所有目标节点的LCA
        情况2：LCA就是节点列表中的某一个节点, 其它所有节点是LCA的子孙节点
    """
    if root is None:
        return None

    if root.val in val_list: # 需要保证所有的目标节点都存在
        return root

    left_ = lowestCommonAncestor(root.left_child, val_list)
    right_ = lowestCommonAncestor(root.right_child, val_list)
    
    if left_ and right_:
        return root

    return left_ if right_ is None else right_
    pass



p_flag = q_flag = False
def lowestCommonAncestor_v3(root, p, q):
    res = lca_find(root, p, q)
    global p_flag, q_flag
    if p_flag and q_flag:
        return res
    else:
        return None

    pass

def lca_find(root, p, q):
    """给一棵元素各不相同的二叉树，求解其中数值为p和q的两个节点的最近公共祖先节点, 如果p或q不存在，那么返回None
    思路：
        情况1：如果一个节点的左右子树分别包含这两个元素，那么该节点就是p和q的LCA
        情况2：LCA就是p或q, 另一个节点是LCA的子孙节点
    """
    if root is None:
        return None

    left_ = lowestCommonAncestor(root.left_child, p, q)
    right_ = lowestCommonAncestor(root.right_child, p, q)
    
    if left_ and right_:
        return root
    
    # 后序位置：这里是从先序遍历的位置调过来的，因为先前假设了p和q一定存在，那么当前检测到的目标节点可以直接返回，另一个节点必然在当前节点的子孙节点中或者父节点的另一个分支中，所以不需要继续遍历当前节点的子孙节点了。但当我们把“p和q一定存在”的假设去掉后，必须要把当前root的所有子孙节点都遍历后，确定另一个节点在不在，才能判断当前节点是否是lca或者只是某一个目标节点
    global p_flag, q_flag
    if root.val in [p, q]: # 需要保证所有的目标节点都存在
        if root.val = p:
            p_flag = True
        if root.val = q:
            q_flag = True
        return root

    # 这里的返回，只能说某一侧分支内含有目标节点，但不一定全，需要结合标志一起判断 
    return left_ if right_ is None else right_
    pass

def lowestCommonAncestor_v3(root, p, q):
    """给你一个不含重复数值的BST, 以及存在于树中的两个节点p和q, 计算它们的LCA.
    思路：按照先前的思路，LCA要么是p和q中的一个，要么p和q分布在LCA两侧分支。BST保证了节点中序遍历的有序性，所以可以通过数值比较代替局部遍历。
    """
    min_n = min(p, q)
    max_n = max(p, q) # p+q-min_n
    return lca_find_v2(root, min_n, max_n)

def lca_find_v2(root, min_node, max_node):
    if root is None:
        return None

    # if root.val in [min_node, max_node]:
        # return root

    # if root.val > min_node and root.val < max_node:
        # return root

    if root.val < min_node:
        return lca_find_v2(root.right_child, min_node, max_node)

    if root.val > min_node:
        return lca_find_v2(root.left_child, min_node, max_node)
    
    # root.val 在[min_n, max_n]区间上
    return root
    pass

def lowestCommonAncestor_v3(p, q):
    """给定一个二叉树，树中每个node都包含指向父节点的指针，给出p和q两个node, 求解LCA.
    思路：求解两个单链表的最近交叉点
    """
    
    if p == q and p is not None:
        return p

    p = p.parent
    q = q.parent

    while(p and q):
        if p == q:
            return p
        p = p.parent
        q = q.parent
    if p is None or q is None:
        return None
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
    # print(leveltravers(Tree.root))
    # ans = tree_Duplicated_subtrees(Tree.root)
    # print(list(ans))

    # =============测试归并排序应用1 =======================
    # nums = [5,4,2,3,6,1]
    # mergeCount(nums)
    # =============测试归并排序应用2:翻转对 =======================
    # nums = [4,1,7,3,2,3,1]
    # print(MergeSort_app2(nums))
    # print(flip_pair_count)
    # =============测试归并排序应用3:区间和的个数 =======================
    # nums = [-2,5,-1]
    # lower = -2
    # upper = 2
    # countrangesum(nums, lower, upper)
    pass
