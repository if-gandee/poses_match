## 连续pose匹配，用于舞蹈打分等场景

### 1. 需求描述
给定一个参考pose序列，从另一段pose序列中找到与关键帧最匹配的pose序列

###2. 算法原理
 * 选定一个参考关键帧Si
 * 根据时间窗口或是帧序号窗口将n个候选目标帧（Tj、Tj+1…Tj+n）与Si配对，每个配对作为一个节点。
 * 根据时间帧序列连接节点建立有向权重图，此外还需要约束每个关键帧配对的目标帧的帧序号（或时间戳）递增。每条边的权重大小为其实节点所表示的match计算两pose得到的loss
 * 求全局最优解，得到的最短路径即为每个参考帧目标帧对