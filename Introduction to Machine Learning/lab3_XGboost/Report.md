## **Report**
---
### **XGBoost**
PB20051035贺维易
### **1.实验原理**
---
#### **1.1 理论推导**
  
由于在任务文档里已经详细给出了XGBoost的实验原理，因此这里不再赘述。我们仅给出最优的权重和这棵决策树的得分：

每个叶子结点的最优权重：
$$
w_j^{*} = -\frac{G_j}{H_j+\lambda}
$$
这棵决策树的得分：
$$
Obj = -\frac{1}{2} \sum_{j = 1}^{T} \frac{G_j^2}{H_j + \lambda} + \gamma T
$$
其中，$G_j$是叶子结点$j$所包含样本的一阶偏导数累加之和，$H_j$是叶子结点$j$所包含样本的二阶偏导数累加之和，$T$为叶子数。

#### **1.2 停止策略选择**

对于如何决定一个节点是否还需要继续划分，我们设定如下的停止策略：
  
- 设定树的最大深度阈值max_depth，如果树的深度大于阈值则停止划分；
- 设定阈值min_child_sample，以限制划分后叶子结点的样本数不能小于这个值。

对于整个算法如何终止，我们设定如下的停止策略：

- 设定决策树的最大棵树$M$，使得算法在学习$M$棵决策树后停下来。
  
#### **1.3 树的存储结构**
我们选用一个字典来存储每一棵树的结构。

### **2.实验步骤**
---
- 读入数据并划分数据集

**训练XGBoost模型**
- 初始化预测值$\hat y$
- 计算$G$和$H$
- 根据$G$和$H$构建一棵新的决策树
- 对预测值做出更新
- 对以上步骤循环直到达到最大训练棵树

**决策树构造过程**
- 遍历所有特征及对应的划分点
- 计算划分后的$G$和$H$
- 计算划分后的增益变化并找到最大增益对应的划分点
- 计算叶子最优权重
- 记录树的结构并递归建树
### **3.部分代码实现**
---
- **3.1切分数据集**

选择75%作为训练集，25%作为测试集
```python
row = df.shape[0]
col = df.shape[1]
k = int(0.75*row)
train = df[0:k]
test = df[k:row]
X_train = train.iloc[:,0:col-1]
y_train = train.iloc[:,col-1]
X_test = test.iloc[:,0:col-1]
y_test = test.iloc[:,col-1]
```
- **3.2计算一阶和二阶偏导**
```python
def _cal_grad(self, y_hat, Y):
        return 2* (y_hat - Y)

def _cal_hess(self,y_hat, Y):
        return np.array([2]*Y.shape[0])
```

- **3.3训练XGBoost模型**

我们也绘制了Loss曲线。
```python
def fit(self, X, Y):
    X = X.reset_index(drop='True')
    Y = Y.values
    # 将base_score设为Y的均值
    self.base_score = np.mean(Y)
    y_hat = np.array([self.base_score]*Y.shape[0])
    for t in range(self.M):
            
        X['g'] = self._cal_grad(y_hat, Y)
        X['h'] = self._cal_hess(y_hat, Y)
            
        f_t = pd.Series([0]*Y.shape[0])
        self.tree_structure[t+1] = self._build_tree(X, f_t, 1)

        y_hat = y_hat + f_t # 对预测值更新
        error = np.sum((y_hat - Y)**2)
        error_table.append(error)
        
        if self.plot:
            from matplotlib import pyplot as plt
            plt.title('error ratio of training')
            plt.xlabel('Tree Number')
            plt.ylabel('Error')
            x = list(range(self.M+1))
            plt.plot(x, error_table)
            plt.show()
```

- **3.4构造决策树**

我们将决策树的算法放入类XGBoost成为其中的一个函数，这只是为了调用时更为方便，事实上，决策树算法与XGBoost算法仍是明显区分开的。

下面仅给出关键步骤：
```python
for feature in [x for x in X.columns if x not in ['g','h','y']]: # 遍历所有特征
    for f_value in list(set(X[feature])): # 遍历对应特征的所有划分点
                
        # 如果分裂后左右样本数目都小于指定值则退出
        if self.min_sample:
            if (X.loc[X[feature] < f_value].shape[0] < self.min_sample)\
                |(X.loc[X[feature] >= f_value].shape[0] < self.min_sample):
                continue
                
        # 计算划分后对应的一阶导和二阶导
        G_left = X.loc[X[feature] < f_value,'g'].sum()
        G_right = X.loc[X[feature] >= f_value,'g'].sum()
        H_left = X.loc[X[feature] < f_value,'h'].sum()
        H_right = X.loc[X[feature] >= f_value,'h'].sum()
               
        # 计算某次分裂带来的增益
        gain = G_left**2/(H_left + self.lambd) + \
                G_right**2/(H_right + self.lambd) - \
                (G_left + G_right)**2/(H_left + H_right + self.lambd)
        gain = gain/2 - self.gamma
        if gain > max_gain:
            best_feature, best_f_value = feature, f_value
            max_gain = gain
            G_left_best, G_right_best, H_left_best, H_right_best = G_left, G_right, H_left, H_right
```
```python
def _get_tree_node_w(self, X, tree, w):
    # 把权重赋给w
    if not tree is None:
        k = list(tree.keys())[0]
        feat,f_value = k[0],k[1]
        X_left = X.loc[X[feat] < f_value]
        id_left = X_left.index.tolist()
        X_right = X.loc[X[feat] >= f_value]
        id_right = X_right.index.tolist()
        for kk in tree[k].keys():
            if kk[0] == 'left':
                tree_left = tree[k][kk]
                w[id_left] = kk[1]
            elif kk[0] == 'right':
                tree_right = tree[k][kk]
                w[id_right] = kk[1]
        
            self._get_tree_node_w(X_left, tree_left, w)
            self._get_tree_node_w(X_right, tree_right, w)
```
- **3.5 Predict and Score**
```python
def predict(self, X):
    X = X.reset_index(drop='True')
    Y = pd.Series([self.base_score]*X.shape[0])

    for t in range(self.M):
        tree = self.tree_structure[t+1]
        y_hat = pd.Series([0]*X.shape[0]) # 初始化
        Y = Y + y_hat
            
    return Y

y_pred = model.predict(X_test)
delta = np.array(y_test) - np.array(y_pred)
delta_norm = np.linalg.norm(delta)
RMSE = delta_norm / sqrt(X_test.shape[0])
```
### **4.实验结果**
---
- **不同参数比较**

75%训练集，25%测试集。

$M$：树的棵数

max_depth：树的最大深度

M | max_depth | lambda | RMSE
--|-----------|--------|-----
3 | 2 | 1 | 0.00025155
5 | 3 | 1 | 0.00021324
5 | 3 | 0.1 | 0.00021474
10 | 3 | 1 | 0.00019783

- Loss函数可视化

我们仅展示最佳模型的Loss图像：


![](https://raw.githubusercontent.com/hwyii/USTC-homework/main/Introduction%20to%20Machine%20Learning/images/XGBoost.png)
  


### **5.实验分析**

- 当树的数量到达 5 棵时，实际上损失已经降到较小，此时增加树的棵数 loss 下降也不多。
- 当对树的棵树、最大深度等参数调整时，RMSE变化较小，这可能是因 XGBoost 本身效果很好，对于树的各种属性的选择都能达到很好的结果，以至于RMSE相差极小，不过仍能看出当树的数量增加时将会有更好的效果。
