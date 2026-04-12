# 鸢尾花书 (Visualize-ML) 参考指南

> 鸢尾花书是一套中文数学可视化教程，通过 Python 代码和矢量图解释数学概念。
> GitHub: [Visualize-ML](https://github.com/Visualize-ML)
> 
> 更新日期: 2026-04-12

---

## 系列总览

| 书名 | 仓库 | 难度 | 核心内容 |
|------|------|------|----------|
| 《编程不难》 | [Book1](https://github.com/Visualize-ML/Book1_Python-For-Beginners) | ⭐ 入门 | Python编程基础 |
| 《可视之美》 | [Book2](https://github.com/Visualize-ML/Book2_Beauty-of-Data-Visualization) | ⭐ 入门 | 数据可视化审美的 |
| **《数学要素》** | [Book3](https://github.com/Visualize-ML/Book3_Elements-of-Mathematics) | ⭐⭐ 基础 | 数学基础全覆盖 |
| **《矩阵力量》** | [Book4](https://github.com/Visualize-ML/Book4_Power-of-Matrix) | ⭐⭐⭐ 进阶 | 线性代数核心 |
| **《统计至简》** | [Book5](https://github.com/Visualize-ML/Book5_Essentials-of-Probability-and-Statistics) | ⭐⭐ 进阶 | 概率统计 |
| 《数据有道》 | [Book6](https://github.com/Visualize-ML/Book6_First-Course-in-Data-Science) | ⭐⭐⭐ 进阶 | 数据科学 |
| 《机器学习》 | [Book7](https://github.com/Visualize-ML/Book7_Visualizations-for-Machine-Learning) | ⭐⭐⭐ 进阶 | 机器学习可视化 |

---

## 《矩阵力量》章节索引 (Book4)

> 适合：线性代数强化、深度学习数学基础
> 难度：⭐⭐⭐
> 
> GitHub: https://github.com/Visualize-ML/Book4_Power-of-Matrix

| 章节 | 主题 | Wiki对应 |
|------|------|----------|
| Ch01 | 向量 | [[1_向量与矩阵]] |
| Ch02 | 向量运算 | [[2_矩阵运算]] |
| Ch03 | 向量范数 | [[5_范数与距离]] |
| Ch04 | 矩阵 | [[1_向量与矩阵]] |
| Ch05 | 矩阵乘法 | [[2_矩阵运算]] |
| Ch06 | 分块矩阵 | [[6_特殊矩阵]] |
| Ch07 | 向量空间 | （待补充） |
| Ch08 | 几何变换 | （待补充） |
| Ch09 | 正交投影 | （待补充） |
| Ch10 | 数据投影（PCA等） | [[4_矩阵分解]] |
| Ch11 | 矩阵分解 | [[4_矩阵分解]] |
| Ch12 | Cholesky分解 | [[4_矩阵分解]] |
| Ch13 | 特征值分解 | [[3_特征值与特征向量]] |
| Ch14 | 深入特征值分解 | [[3_特征值与特征向量]] |
| Ch15 | 奇异值分解(SVD) | [[4_矩阵分解]] |
| Ch16 | 深入奇异值分解 | [[4_矩阵分解]] |
| Ch17 | 多元函数微分 | [[1_导数与微分]] [[2_链式法则]] |
| Ch18 | 拉格朗日乘子法 | （待补充） |
| Ch19 | 直线到超平面 | （待补充） |
| Ch20 | 圆锥曲线 | （待补充） |
| Ch21 | 曲面和正定性 | [[6_特殊矩阵]] |
| Ch22 | 数据与统计 | [[3_期望与方差]] |
| Ch23 | 数据空间 | （待补充） |
| Ch24 | 数据分解 | [[4_矩阵分解]] |
| Ch25 | 数据应用 | （待补充） |

### 核心亮点

- **Ch09 正交投影**：最小二乘法的几何解释
- **Ch10 数据投影**：PCA/SVD在数据降维中的应用
- **Ch15/16 SVD**：深入理解奇异值分解
- **Ch17 多元函数微分**：梯度、Hessian矩阵在机器学习中的应用
- **Ch21 曲面和正定性**：Hessian正定性判断极值

---

## 《统计至简》章节索引 (Book5)

> 适合：概率论强化、机器学习统计基础
> 难度：⭐⭐
> 
> GitHub: https://github.com/Visualize-ML/Book5_Essentials-of-Probability-and-Statistics

| 章节 | 主题 | Wiki对应 |
|------|------|----------|
| Ch02 | 统计描述 | [[1_概率基础]] |
| Ch03 | 古典概型 | [[1_概率基础]] |
| Ch04 | 离散随机变量 | [[2_随机变量]] |
| Ch05 | 离散分布 | [[4_重要分布]] |
| Ch06 | 连续随机变量 | [[2_随机变量]] |
| Ch07 | 连续分布 | [[4_重要分布]] |
| Ch08 | 条件概率 | [[1_概率基础]] |
| Ch09 | 一元高斯分布 | [[4_重要分布]] |
| Ch10 | 二元高斯分布 | [[4_重要分布]] |
| Ch11 | 多元高斯分布 | [[4_重要分布]] |
| Ch12 | 条件高斯分布 | [[2_随机变量]] |
| Ch13 | 协方差矩阵 | [[3_期望与方差]] |
| Ch14 | 再谈随机变量 | [[2_随机变量]] |
| Ch15 | 蒙特卡洛模拟 | [[3_积分]] |

### 核心亮点

- **Ch11 多元高斯分布**：深度学习中几乎所有关于不确定性的基础
- **Ch12 条件高斯分布**：线性高斯系统的贝叶斯推断核心
- **Ch13 协方差矩阵**：协方差矩阵的几何意义和性质
- **Ch15 蒙特卡洛模拟**：变分推断、RL中的梯度估计

---

## 《数学要素》章节索引 (Book3)

> 适合：数学基础薄弱者的系统复习
> 难度：⭐⭐ 入门到进阶
> 
> GitHub: https://github.com/Visualize-ML/Book3_Elements-of-Mathematics

| 章节 | 主题 | Wiki对应 |
|------|------|----------|
| Ch01 | 万物皆数 | — |
| Ch02 | 乘除 | — |
| Ch03 | 几何 | — |
| Ch04 | 代数 | — |
| Ch05 | 笛卡尔坐标系 | — |
| Ch06 | 三维坐标系 | — |
| Ch07 | 距离 | [[5_范数与距离]] |
| Ch08-09 | 圆锥曲线 | — |
| Ch10 | 函数 | — |
| Ch11 | 代数函数 | — |
| Ch12 | 超越函数 | — |
| Ch13 | 二元函数 | [[1_导数与微分]] |
| Ch14 | 数列 | — |
| Ch15 | 极限与连续 | — |
| Ch16 | 导数 | [[1_导数与微分]] |
| Ch17 | 微分中值定理 | — |
| Ch18 | 导数的应用 | — |
| Ch19 | 泰勒公式 | — |
| Ch20 | 积分 | [[3_积分]] |

---

## 如何使用鸢尾花书

### 学习路径建议

```
数学基础薄弱
    ↓
《数学要素》(Book3) 打下基础
    ↓
《矩阵力量》(Book4) 线性代数
+ 《统计至简》(Book5) 概率统计
    ↓
Wiki数学基础 + 鸢尾花书代码实战
```

### 代码实战

每个章节都有配套的 Python 代码：

```bash
# 克隆 Book4 代码
git clone https://github.com/Visualize-ML/Book4_Power-of-Matrix.git

# 查看 SVD 章节代码
cd Book4/Book4_Ch15_Python_Codes/
```

### PDF 讲义

每个章节都有配套 PDF 讲义，可在 GitHub releases 或直接下载：

```
Book4_Ch15_奇异值分解__矩阵力量__从加减乘除到机器学习.pdf
```

---

## 在Wiki中的应用

鸢尾花书的优势在于**几何直觉 + Python 代码**，非常适合补充 Wiki 中缺少的可视化部分。建议：

1. 学习某个概念时，先看鸢尾花书的 PDF 讲义建立几何直觉
2. 再看 Wiki 中的公式推导加深理解
3. 最后运行 Python 代码亲手实践

---

## Tags

#鸢尾花书 #visualize-ml #linear-algebra #probability #statistics #python #visualization
