# 矩阵理论参考资源

> 深度学习与大模型相关的矩阵理论参考文献整理
> 更新日期: 2026-04-12

---

## 核心必读

### 1. The Matrix Calculus You Need For Deep Learning

> Terence Parr, Jeremy Howard — **最推荐的深度学习矩阵微积分入门**

| 属性 | 内容 |
|------|------|
| **作者** | Terence Parr (University of San Francisco), Jeremy Howard (fast.ai) |
| **arXiv** | [arXiv:1802.01528](https://arxiv.org/abs/1802.01528) |
| **GitHub** | [GitHub](https://github.com/parrt/lolviz) |
| **类型** | 教程/笔记 |
| **难度** | ⭐⭐ 入门 |

#### 核心内容

本文档讲解深度学习中所需的矩阵微积分，从标量求导出发，逐步扩展到向量和矩阵。

**核心章节：**

1. ** Preliminaries ** — 约定与表示法
2. **Gradients of Functions** — 函数梯度
3. **Gradients of Matrices** — 矩阵梯度
4. **Derivatives with Matrices** — 矩阵形式的导数
5. **Neural Networks** — 在神经网络中的应用

**关键公式：**

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \cdots \\ \frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

#### 为什么推荐

- 从**标量→向量→矩阵**循序渐进
- 所有公式都有**完整推导**
- 专门针对**深度学习**场景
- 作者 Jeremy Howard 是 fast.ai 创始人

#### 阅读建议

建议配合 **Matrix Cookbook** 一起阅读，前者偏计算，后者偏公式速查。

---

### 2. The Matrix Cookbook

> Kaare Brandt Petersen, Michael Syskind Pedersen — **矩阵公式速查手册**

| 属性 | 内容 |
|------|------|
| **作者** | Kaare Brandt Petersen, Michael Syskind Pedersen |
| **最新版本** | v2 (Nov 2012) |
| **下载地址** | [PDF (KIT](https://www.math.uwaterloo.ca/~wgilbert/31/HarderWilkiem.pdf)) / [PDF (Petersen site)](https://www.math.uwaterloo.ca/~wgilbert/31/HarderWilkiem.pdf) |
| **GitHub** | [TheMatrixCookbook](https://github.com/ctwardy/TheMatrixCookbook) |
| **类型** | 公式手册 |
| **页数** | ~60 pages |

#### 核心内容

**目录结构：**

1. **基础** — 记号、定义、基本操作
2. **代数** — 逆矩阵、转置、迹、行列式
3. **导数** — 标量/向量/矩阵函数的导数
4. **积分** — 矩阵函数的积分
5. **特殊矩阵** — Kronecker积、Hadamard积等
6. **概率与统计** — 多元正态分布相关公式
7. **矩阵恒等式** — Sherman-Morrison、Woodbury等

**核心公式速查：**

| 场景 | 公式 |
|------|------|
| $\frac{\partial \mathbf{a}^T\mathbf{x}}{\partial \mathbf{x}}$ | $\mathbf{a}$ |
| $\frac{\partial \mathbf{x}^T\mathbf{A}\mathbf{x}}{\partial \mathbf{x}}$ | $(\mathbf{A} + \mathbf{A}^T)\mathbf{x}$ |
| $\frac{\partial \mathbf{A}^{-1}}{\partial \alpha}$ | $-\mathbf{A}^{-1} \frac{\partial \mathbf{A}}{\partial \alpha} \mathbf{A}^{-1}$ |
| $\frac{\partial \det(\mathbf{A})}{\partial \mathbf{A}}$ | $\det(\mathbf{A}) \cdot (\mathbf{A}^{-1})^T$ |

#### 使用建议

- 作为**速查手册**，不需要通读
- 需要矩阵导数时直接翻到对应章节
- 注意不同作者的符号约定可能不同

---

## 进阶参考

### 3. Linear Algebra Done Right (Axler)

> Sheldon Axler — **线性代数理论进阶**

| 属性 | 内容 |
|------|------|
| **作者** | Sheldon Axler (San Francisco State University) |
| **出版社** | Springer (Undergraduate Texts in Mathematics) |
| **难度** | ⭐⭐⭐ 进阶 |

#### 特点

- 不讲行列式直到最后一章（争议但独特）
- 强调**线性算子**和**特征值**的几何意义
- 证明优雅简洁

#### 适合读者

- 已有线性代数基础，想深入理解理论
- 想要理解为什么特征值如此重要
- 研究生水平

---

### 4. Matrix Analysis (Horn & Johnson)

> Roger A. Horn, Charles R. Johnson — **矩阵分析领域经典**

| 属性 | 内容 |
|------|------|
| **作者** | Roger A. Horn, Charles R. Johnson |
| **出版社** | Cambridge University Press |
| **难度** | ⭐⭐⭐⭐ 高级 |

#### 核心内容

- 特征值不等式
- 正定矩阵
- 矩阵范数
-扰动理论

#### 适合读者

- 从事矩阵理论研究
- 需要深入理解收敛性分析
- 研究生/研究者

---

### 5. Gilbert Strang — 线性代数系列

| 资源 | 类型 | 难度 |
|------|------|------|
| [MIT 18.06](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) | 视频课程 | ⭐⭐ |
| [Linear Algebra](https://math.mit.edu/~gs/linearalgebra/) | 教材 | ⭐⭐ |
| [Computational Science](https://math.mit.edu/~gs/cse/) | 进阶 | ⭐⭐⭐ |

#### 特点

- 实践导向，每个概念都有具体应用
- MIT OpenCourseWare 有完整视频
- 非常适合工程背景

---

## 深度学习专项

### 6. Mathematics for Machine Learning (2020)

> Marc Deisenroth, Aldo Faisal, Cheng Soon Ong — **机器学习数学基础**

| 属性 | 内容 |
|------|------|
| **出版社** | Cambridge University Press |
| **PDF** | [Free PDF](https://mml-book.github.io/) |
| **GitHub** | [mml-book.github.io](https://github.com/mml-book/mml-book.github.io) |
| **难度** | ⭐⭐ 中级 |

#### 核心章节

1. **线性代数** — 向量空间、矩阵、特征分解
2. **解析几何** — 内积、范数、投影
3. **矩阵分解** — SVD、特征分解、LU
4. **向量微积分** — 梯度、Jacobian、Hessian
5. **概率与分布** — 多元高斯、贝叶斯

---

## 快速跳转

| 需求 | 推荐资源 |
|------|----------|
| 深度学习矩阵求导入门 | [[#1 The Matrix Calculus You Need For Deep Learning]] |
| 矩阵公式速查 | [[#2 The Matrix Cookbook]] |
| 理解特征值几何意义 | [[#3 Linear Algebra Done Right]] |
| 矩阵理论深入研究 | [[#4 Matrix Analysis]] |
| 系统学习线性代数 | [[#5 Gilbert Strang 课程]] |
| 机器学习数学全貌 | [[#6 Mathematics for Machine Learning]] |

---

## Tags

#matrix-calculus #linear-algebra #deep-learning #reference #pdf
