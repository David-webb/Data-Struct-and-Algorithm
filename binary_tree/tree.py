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
        self.next = None
        pass

class tree():
    def __init__(self, root_data=None):
        self.root = node(root_data)
        # self.maxtreedepth = 0
        pass

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
        # rt = self.root
        # ans = []
        if pnode:
            # print(pnode.val)
            ans.append(pnode.val)
            if pnode.left_child:
                self.preOrderSearch(pnode.left_child, ans)
            if pnode.right_child:
                self.preOrderSearch(pnode.right_child, ans)
        return ans
        pass

    def InOrderSearch(self, pnode):
        if pnode:
            if pnode.left_child:
                self.InOrderSearch(pnode.left_child)

            print(pnode.val)

            if pnode.right_child:
                self.InOrderSearch(pnode.right_child)
        pass

    def PostOrderSearch(self):
        if pnode:
            if pnode.left_child:
                self.PostOrderSearch(pnode.left_child)
            if pnode.right_child:
                self.PostOrderSearch(pnode.right_child)
            print(pnode.val)
        pass


def maxdepth(node, lastdepth=0):
    """二叉树的最大深度：所谓最大深度就是根节点到「最远」叶子节点的最长路径上的节点数
    """
    depth = lastdepth + 1 if node else lastdepth
    lnode, rnode = node.left, node.right
    # while(lnode or rnode):
        # depth += 1
        # lnode 
    ldp = rdp = -1
    if lnode:
        ldp = maxdepth(lnode, depth)
    if rnode:
        rdp = maxdepth(rnode, depth)
    return max(ldp, rdp, depth)

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
    if not root:
        return 0
    
    lmaxdep = rmaxdep = 0
    # l_dia = r_dia = 0
    lmaxdep = maxdepth(node.left, 0)
    rmaxdep = maxdepth(node.right, 0)
    tmp_dia = lmaxdep + rmaxdep + 2
    global max_diameter
    ans = max(max_diameter, tmp_dia)
    return ans 
    
def leveltravers(root):
    """层序遍历"""
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
    """二叉树转链表(inplace)"""
    tree_squeeze2linked(root)
    left2right(root)
    return root
    

def left2right(root):
    """将左孩子统一转到右孩子位置"""
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
    """将树转化成链表（限制为in-place操作）"""
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
            rt.next = buffer[0]
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
    pre_l = [i for i in preOrder if i in in_l]
    pre_r = [j for j in preOrder if j in in_r]
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

if __name__ == "__main__":
    # ========= 测试 go_linkedsqueezing函数 ============
    # 构建树（注意节点的val数值要和题目一致!）
    Tree = build_tree()
    root = go_linkedsqueezing(Tree.root)
    print(leveltravers(Tree.root))
    print(Tree.preOrderSearch(Tree.root))

    # ============测试：tree_Duplicated_subtrees ============
    Tree = build_tree(is_full_binaryTree=False)
    print(leveltravers(Tree.root))
    ans = tree_Duplicated_subtrees(Tree.root)
    print(list(ans))

