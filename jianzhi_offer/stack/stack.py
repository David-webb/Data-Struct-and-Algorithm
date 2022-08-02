# -*- coding:utf-8 -*-
""" 
    title:用两个栈实现队列
    用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
"""
class buildQueyeWithStack():
    def __init__():
        self.s1 = [] 
        self.s2 = [] 
        pass

    def push(s):
        self.s1.append(s)

    def pop():
        while(self.s1):
            self.s2.append(self.s1.pop())

        res = self.s2.pop()

        while(self.s2):
            self.s1.append(self.s2.pop())

        return res



"""
    title: 栈的压入弹出序列
    要求：
        输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假
    设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1是该压栈序列
    对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度
    是相等的）
    思路：
        思路：借用一个辅助的栈，遍历压栈顺序，先将第一个放入栈中，这里是1，然后判断栈顶元素是不
    是出栈顺序的第一个元素，这里是4，很显然1≠4，所以我们继续压栈，直到相等以后开始出栈，出栈一
    个元素，则将出栈顺序向后移动一位，直到不相等，这样循环等压栈顺序遍历完成，如果辅助栈还不为
    空，说明弹出序列不是该栈的弹出顺序。
"""
def check_right_pop_order(push_ord, pop_ord):
    t_stack = []
    stop_len = len(pop_ord)
    work_p = 0
    for i in push_ord:
        t_stack.append(i)
        if(i == pop_ord[work_p]): # 注意这里这样写是因为有条件：“入栈的所有数字均不相等”
            while(t_stack and work_p < stop_len):
                ti = t_stack.pop() 
                if ti == pop_ord[work_p]:
                    work_p += 1
                else:
                    t_stack.append(ti)
                    break
    
    if work_p == stop_len:
        return True
    else:
        return False
    pass

def check_right_pop_order_v2(push_ord, pop_ord):
    """
        剑指offer给出的代码，更精简
    """
    if not push_ord and not pop_ord:
        return False
    
    t_stack = []
    for i in push_ord:
        t_stack.append(i)
        while(t_stack and t_stack[-1] == pop_ord[0]):
            t_stack.pop()
            pop_ord.pop(0)

    if t_stack:
        return False
    else:
        return True
        
    pass


if __name__ == "__main__":
    # **************测试 “栈的压入弹出序列” ****************
    t1 = [1,2,3,4,5]
    t2 = [4,3,5,2,1]
    print(check_right_pop_order_v2(t1, t2))
    pass
