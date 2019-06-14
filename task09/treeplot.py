#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/6/14 15:11
@Author  : xumj
'''

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="10")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<|-")


# 定义决策树决策结果的属性，用字典来定义，也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}
# 其中boxstyle表示文本框类型，sawtooth是波浪型的，fc指的是注释框颜色的深度
# arrowstyle表示箭头的样式

# 绘制带箭头的注解

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    nodeTxt指要显示的文本，centerPt指的是文本中心点，parentPt指向文本中心的点
    '''
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', \
                            va='center', ha='center', bbox=nodeType, arrowprops \
                                =arrow_args)


# 获取叶节点数目
def getNumLeafs(myTree):
    numleafs = 0
    firststr = list(myTree.keys())[0]
    secondDict = myTree[firststr]

    # 递归调用计算节点数
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 如果节点还嵌套着字典
            numleafs += getNumLeafs(secondDict[key])
        else:
            numleafs += 1

    return numleafs


# 获取树的深度
def getTreeDepth(myTree):
    maxDepth = 0
    firststr = list(myTree.keys())[0]
    secondDict = myTree[firststr]
    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 在父子节点之间填充文本信息
def plotMidText(cntrPt, parentPt, txtString, **args):
    """
    cntrPt: 子节点
    parentPt： 父节点
    txtString:填充信息
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 中点横坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, **args)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firststr = list(myTree.keys())[0]

    # plotTree.totalW
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)  # 计算宽和高
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firststr, cntrPt, parentPt, decisionNode)

    secondDict = myTree[firststr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 减少 y 的偏移

    for key in secondDict:
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key), rotation=20, va='center', ha='center')

    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def retrieveTree(i):
    # 预先设置树的信息
    listOfTree = [{'no surfacing': {0: 'no', 1: {'flipper': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flipper': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                  {'a1': {0: 'b1', 1: {'b2': {0: {'c1': {0: 'd1', 1: 'd2'}}, 1: 'c2'}}, 2: 'b3'}}]
    return listOfTree[i]

myTree = retrieveTree(2)
createPlot(myTree)