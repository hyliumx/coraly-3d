# 珊瑚单体点云标注方法调研报告

日期：2026-11-16 · 用途：决定第一步怎么解决"没有带标签数据"的问题

---

## 0. 结论先说

1. **问题定义**：把场景稠密点云 P 划分成若干子集 {C₁, C₂, …}（每个子集是一株珊瑚，其余是底质或背景），然后给每个子集一个生长型标签。这本质上是**3D 实例分割 + 实例分类**。
2. **目前没有找到现成的"3D 珊瑚单体 + 生长型"公开数据集**（见第 6 节），所以标注方法本身值得做，而且有机会成为研究贡献。
3. **推荐路线**：用"点标注作 prompt → 在原始照片上用 SAM/CoralSCOP 生成 mask → 用 SfM 相机反投影到点云 → 多视角投票"自动生成候选单体，**人只做审核和少量修正**（修正时可以用 Point-SAM / AGILE3D 点几下）。同时用纯人工精标一小块样地当基准，统计每种方法的"每株人工时间"和"与精标的 IoU"。
4. **最紧急的事**：向 Jordan 要**一个带 mask 的样例场景**，并问清楚标注是标在**正射影像上还是原始照片上**。这一点会决定管线的一半设计（见第 7 节问题清单）。

---

## 1. 评价标准（建议先定下来）

| 维度 | 指标 | 说明 |
|---|---|---|
| 效率 | 每株珊瑚的人工时间（分钟/株），以及每平方米的人工时间 | 计时要包括审核和修正，不能只算算法运行时间 |
| 边界正确性 | 实例 IoU、AP50/AP25（以人工精标为准） | 匹配规则：IoU > 0.5 记为同一株 |
| 拆分/合并错误 | 过分割率、欠分割率 | 相邻同种群体粘在一起是珊瑚数据最常见的错误 |
| 类别正确性 | 生长型准确率、混淆矩阵 | 和边界分开统计 |

**基准集**：选 1 块样地（比如 2 m × 2 m，含 50～100 株），全部纯人工精标并记录用时。之后每种方法都在这块样地上比较。参考数据：Sandin 组在 Palmyra 用 Viscore 手工描 *Porites*，**平均每平方米约 1 小时**，而且"修改自动预测结果常常和从零开始画一样慢，因为最耗时的是描边界"（[Frontiers in Robotics and AI 2022](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.884317/pdf)）。这条结论对我们的设计很关键：**自动方法的边界要足够好，否则人机协同反而不省时间。**

---

## 2. Jordan 那边可能的工具链（推测，需要确认）

你 PDF 里引用的 Edwards、Sandin、Petrovic 等人属于 Scripps / 100 Island Challenge 团队。从公开文献看，他们常用的工具链是：

- **Agisoft Metashape** 做 SfM 重建，输出稠密点云、正射影像（orthoprojection）、DEM
- **Viscore**（UCSD 开发）做可视化和标注，可以在点云上点一个点，然后调出看到这个点的原始照片；也支持在点云上用画笔逐点标注
- 群体轮廓（colony delineation）通常是**画在正射影像上**的 2D 多边形，比如 NOAA 的 SfM 群体生长率数据（[data.gov](https://catalog.data.gov/dataset/structure-from-motion-sfm-coral-vital-rates-annotations-derived-from-fixed-site-imagery-ac-20143)）就是在正射投影上描群体边界

如果 Jordan 也用 Metashape + Viscore，那么相机参数可以从 Metashape 导出（cameras.xml，或者直接用它的 Python API），标注数据可能是 Viscore 的导出格式。**这些都只是推测，要问他。**

---

## 3. 三类方法详解

### A. 纯人工（作基准）

| 项 | 内容 |
|---|---|
| 工具 | CloudCompare（Segment 工具，框选/多边形切割）、Viscore（画笔逐点标注）、Metashape（点云分类工具） |
| 输入 | 只要稠密点云（有颜色更好） |
| 适用性 | 一定能做，但有三个难点：① 相邻的同种群体在 3D 里很难分界，专家通常要看原始照片才能判断；② 枝状珊瑚的内部被遮挡，点云有空洞；③ 底质与皮壳状珊瑚的边界几乎没有几何差异，只能靠颜色 |
| 效率 | 约 1 小时/m²（上面 Palmyra 的数据，单一属）。按每平方米十几株粗估，**每株几分钟**（我的估算，需要在基准样地上实测） |
| 角色 | 只用来做基准集和最终修正，不适合做大规模主力 |

### B. 2D → 3D 投影（推荐主线）

核心思路：2D 分割模型（SAM、CoralSCOP）在照片上比任何 3D 模型都成熟得多，而 SfM 天生就有"每张照片的相机参数 + 3D 点云"，可以把 2D 结果搬到 3D 上。珊瑚领域已有先例：

- [Hopkinson et al. 2020, PLOS ONE](https://pmc.ncbi.nlm.nih.gov/articles/PMC7093084)：把 3D 网格的面投影回原始视频帧做分类，多视角融合后约 96% 准确率
- [Pierce et al. 2021, Frontiers in Marine Science](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.706674/pdf)：先在照片上做语义分割，再在 Metashape 里用标签图代替原图重建纹理，把标签投影到模型上，3D 精度约 91%
- [TagLab 团队 2025 年的 poster](https://iris.cnr.it/bitstream/20.500.14243/566983/4/Corsini%20et%20al_Poster-EG%202025.pdf)：用 SAM 在照片上分割，再用 Metashape 相机参数把 mask 投到 3D 模型，并把 3D 中重叠的 mask 合并以免重复计数

**这些都是语义级别（"这里是什么类"）或单个目标的，还没看到有人在珊瑚上做完整的"实例级"版本。**这正是可以做的地方。

#### B1. 点标注作 prompt 的多视角 SAM（主推）

```
稀疏点标注(3D 点 或 正射影像像素→3D)
   │ 用相机参数投影到"能看到它"的多张原始照片（深度测试排除被遮挡的视角）
   ▼
每张照片：SAM / CoralSCOP 以该点为 prompt 生成 mask
   │ 把 mask 反投影到点云（同样需要深度测试）
   ▼
每个 3D 点收集多个视角的投票 → 投票率 > 阈值 的点归入这一株
   │ 不同 prompt 生成的 3D 区域高度重叠 → 合并（同一株上有多个标注点）
   ▼
候选单体 + 置信度 → 人工审核
```

- **方法基础**：[SAMPro3D](https://arxiv.org/html/2311.17707v2) 的做法几乎一样（把 3D prompt 投影到各帧给 SAM，用深度测试处理遮挡，按视角打分过滤，按 3D 表面重叠合并），只是它的 prompt 是随机采样的点，而我们有**专家的点标注**，prompt 的质量高得多，而且自带类别。
- **输入**：原始照片、每张照片的内参（含畸变）和外参、稠密点云；深度测试最好有 mesh（可以从点云用 Metashape 或 Poisson 重建生成），或者直接用点云做 z-buffer 渲染。
- **现成工具**：Metashape Python API 或 COLMAP 读相机；SAM 2 / CoralSCOP 做 2D 分割；Open3D 或 PyTorch3D 做投影和深度渲染。
- **适用性**：高。SfM 数据本来就是从这些照片算出来的，相机和点云天然对齐，不需要额外配准。
- **风险**：① 枝状珊瑚在 SAM 里容易被切碎或和背景粘连，可能需要每个视角给多个 prompt，或改用 CoralSCOP；② 浑水、偏色、焦散光斑；③ 相邻同种群体连在一起时，2D 和 3D 都分不开，必须由人来判断。
- **预计效率**：标注点已经存在时，人工时间只剩审核，**粗估每株几秒到几十秒**（我的估算，需要实测）。

#### B2. 直接用已有的 2D 实例 mask

Jordan 说有"部分实例分割 mask"。如果 mask 标在原始照片上，直接用 B1 后半段（反投影 + 多视角投票）就能得到 3D 单体，这批数据可以作为**第一批训练数据和验证集**。

#### B3. 正射影像路线（最省事的基线）

如果标注标在正射影像上：正射像素 ↔ 世界坐标 XY 是线性对应的（GeoTIFF 带地理变换），DEM 给出 Z。把落在多边形 XY 范围内、且接近 DEM 表面的点选出来，就得到一个单体。

- **优点**：不需要相机参数，一天就能写完。
- **缺点**：正射是俯视的，**侧面、悬垂部分、底下的结构会被漏掉或误归**；你 PDF 里引用的文献也说正射拼接会降低分辨率、边界模糊。适合做 baseline 和快速出第一版数据。

#### B4. 全自动零样本 3D 实例分割（作候选生成器，不作真值）

| 方法 | 要点 | 室内数据上的精度 |
|---|---|---|
| SAM3D | 每帧独立用 SAM，再在 3D 合并 | ScanNet200 AP50 38.6 |
| SAMPro3D | 3D prompt 保证跨视角一致 | ScanNet200 AP50 47.2；ScanNet++ 35.6 |
| Segment3D | 需要训练 | ScanNet++ AP50 22.7 |
| Open3DIS、MV3DIS（2026）等 | 2D mask 与 3D 超点结合 | 类似量级 |

（数据来自 [SAMPro3D 论文](https://arxiv.org/html/2311.17707v2) 的对比表。）

- 都需要"带位姿的 RGB-D 帧"。我们没有深度图，但可以从 mesh 渲染出来。
- 室内 AP50 只有 35～47，到珊瑚上只会更低。**适合给人工审核提供候选，或用来处理没有点标注的大片区域，不适合直接当真值。**

### C. 3D 交互式分割

| 方法 | 要点 |
|---|---|
| [AGILE3D](https://ywyue.github.io/AGILE3D)（ICLR 2024） | 在点云上点击正/负点，可以同时分割多个物体；只在 ScanNet（室内）上训练，展示过户外 KITTI-360 但没有定量结果；代码开源 |
| [Point-SAM](https://point-sam.github.io)（ICLR 2025） | "3D 版 SAM"，点 prompt；训练数据更多，作者报告零样本迁移优于 AGILE3D；有代码和在线 demo |

- **输入**：只要点云，不需要照片和相机。
- **适用性**：中等，需要实测。两者都没见过珊瑚，领域差距大；而且需要先把场景切成局部小块。最大的短板是**看不到原始照片的高分辨率纹理**，而专家判断边界恰恰依赖照片。
- **用法建议**：作为 B1 结果的**修正工具**（某株切错了，点几下修好），而不是主力。
- **预计效率**：论文里室内物体通常几次点击就能达到 IoU 0.8 以上，珊瑚上要实测。

### D. 候选生成与人机分工

1. **聚类生成候选**：先用语义模型或颜色/高度把底质去掉，再用超体素（supervoxel）、基于法向和颜色的区域生长、DBSCAN 等做过分割。人只需要"合并碎片"，比"从头描边界"快。可以和 B1 结合：B1 的投票结果按超体素平滑，边界会更干净。
2. **"你切形状、专家选标签"**：
   - 你（或者算法 + 你审核）负责把每株切出来，这一步不需要珊瑚分类的专业知识；
   - 系统自动把每株投影到看得最清楚的 3～5 张原始照片，裁出截图；
   - 专家只看截图选生长型，**每株几秒**。这和 Viscore 里"点一个点 → 调出原始照片"的习惯一致，专家应该容易接受。
   - 如果这株里已经有稀疏点标注，直接多数投票定类别，只把**有冲突或没有点标注的**交给专家。
3. **主动学习（后续）**：用前几批结果训练一个 3D 实例分割模型，让它给新场景出候选，人只审核低置信度的部分。

---

## 4. 方法对比总表

| 方法 | 需要的数据 | 每株人工时间（粗估） | 边界质量 | 实现难度 | 角色 |
|---|---|---|---|---|---|
| A 纯人工 | 点云 | 几分钟 | 最好（但慢） | 无 | 基准集、最终修正 |
| B1 点 prompt + 多视角 SAM | 照片 + 相机 + 点云 + 点标注 | 几秒～几十秒（审核） | 待测，预计较好 | 中 | **主线** |
| B2 已有 2D mask 反投影 | 照片 + 相机 + 点云 + mask | 接近 0 | 取决于 mask | 中（和 B1 共用代码） | 第一批数据 |
| B3 正射 + DEM | 正射 + DEM + 点云 + 2D 标注 | 接近 0 | 俯视部分好，侧面差 | 低 | 快速基线 |
| B4 零样本 3D 实例 | 照片 + 相机 + 深度 | 0（但要大量修正） | 较差 | 中～高 | 无标注区域的候选 |
| C AGILE3D / Point-SAM | 点云 | 待测 | 待测 | 中（要做 UI） | 修正工具 |
| D 人机分工 | 上述任意 + 截图 | 专家每株几秒 | 取决于切分 | 低 | 定类别 |

**"粗估"一列是我根据文献和经验推的，没有珊瑚上的实测数据，正需要通过基准实验来验证。**

---

## 5. 推荐方案（分阶段）

| 阶段 | 做什么 | 产出 |
|---|---|---|
| 0 | 拿到 1 个样例场景；用 CloudCompare 纯人工精标一块样地，记录每株用时 | 基准集 + 人工基线时间 |
| 1 | B3 正射路线 或 B2 已有 mask 反投影（看 Jordan 的标注标在哪里） | 第一版训练数据；和基准比 IoU |
| 2 | B1 点 prompt 多视角 SAM/CoralSCOP + 超体素平滑 | 候选单体 + 置信度 |
| 3 | 审核界面：多视角截图 + 一键接受/拒绝；切错的用 Point-SAM 点击修正 | 修正后的数据；记录每株审核用时 |
| 4 | 在基准上比较 A / B3 / B1 / B1+C 的"时间—精度"曲线 | 可以写进论文的方法比较 |

第 4 步的比较本身就是一个研究贡献：**"在 SfM 珊瑚数据上，用多少人工时间能换到多少实例精度"**，目前文献里没有这样的比较。

---

## 6. 公开数据集（备选）

搜索后**没有找到同时满足"3D 点云 + 单株实例 + 生长型标签"的公开数据集**。比较接近的有：

| 数据 | 内容 | 能用来做什么 | 局限 |
|---|---|---|---|
| [Zenodo 14249838](https://zenodo.org/records/14249838) | 珊瑚礁点云 + KPConv 代码，含 `Labeled_PtClouds.zip`（10.5 GB），CC BY 4.0 | 3D 分割的预训练和测试 | 标签类型（语义还是实例）页面没写清，需要下载确认 |
| [NOAA SfM 群体生长率数据](https://catalog.data.gov/dataset/structure-from-motion-sfm-coral-vital-rates-annotations-derived-from-fixed-site-imagery-ac-20143) | 正射影像上描的群体边界，Metashape 重建，含属/种 | 和 Jordan 的数据形态最接近；可以测 B3 路线 | 只有属/种，没有生长型；页面上没有可下载文件，可能要联系 NOAA |
| Palmyra 数据（[Frontiers 2022](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.884317/pdf)） | Viscore 手工逐点标注的 *Porites*，2013–2020 年 | 3D 真值，适合验证投影方法 | 只有一个属，语义不是实例；"按需提供"。**很可能就是 Jordan 所在团队的数据，值得直接问他** |
| [Hopkinson 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7093084) | 3D 网格 + 多视角分类 | 多视角投影方法的参考 | 语义级 |
| [DeepReefMap 示例数据](https://zenodo.org/record/10624794) | GoPro 视频 + 语义分割模型 | 2D 语义分割预训练 | 没有 3D 实例标签 |
| CoralSCOP / CoralMask、Coralscapes、CoralVOS | 大量 2D 珊瑚 mask | 2D 分割模型（B1 的核心组件） | 只有 2D |

---

## 7. 要问 Jordan 的问题

**最优先（决定管线设计）**

1. 能否先给**一个带标注的样例场景**（照片、相机参数、稠密点云、正射/DEM、mask、点标注都要），大小无所谓，能跑通流程就行？
2. 稀疏点标注和实例 mask 是**标在正射影像上、原始照片上，还是直接标在点云上**？
3. SfM 用的是什么软件（Metashape、COLMAP、其他）？能否导出**每张照片的内参（含畸变）和外参**？如果是 Metashape，能否提供项目文件（.psx）或 cameras.xml？
4. 标注用的是什么工具（Viscore、TagLab、CoralNet、其他），导出格式是什么？

**类别与标注规范**

5. 点标注的类别体系是什么（属/种？还是已经是生长型？）有没有类别表？
6. 生长型分哪几类（枝状、块状、皮壳状、板状、叶状、柱状、蘑菇状……）？由谁定，标准是什么？
7. 实例 mask 是怎么来的（手工、SAM 辅助）？覆盖了场景的多大比例？有没有漏标的群体？
8. 相邻的同种群体怎么定义边界？部分死亡的群体还算一株吗？

**数据量与形式**

9. 总共有多少个场景/样地，每个多大（m²），大约多少株？有没有同一地点不同年份的重复调查？
10. 稠密点云的密度、坐标系、单位；是否有比例尺或地面控制点？
11. 原始照片的分辨率和数量；是否做过颜色校正？

**合作与资源**

12. Frontiers 2022 那篇 Palmyra 的 Viscore 逐点标注数据，我们能用吗？
13. 有没有人（他或学生）可以做几十株的精标，作为基准集？或者能否请专家花一点时间看截图选生长型？
14. 数据使用和发表上有什么限制？

---

## 参考来源

- Frontiers in Robotics and AI 2022（Palmyra，Viscore 手工标注，1 h/m²）：https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.884317/pdf
- Pierce et al. 2021：https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.706674/pdf
- Hopkinson et al. 2020：https://pmc.ncbi.nlm.nih.gov/articles/PMC7093084
- TagLab：https://taglab.isti.cnr.it ；EG 2025 poster：https://iris.cnr.it/bitstream/20.500.14243/566983/4/Corsini%20et%20al_Poster-EG%202025.pdf
- TagLab / CoralNet / Viscore 互操作（ISPRS 2024）：https://isprs-archives.copernicus.org/articles/XLVIII-2-2024/327/2024/
- CoralSCOP-LAT：https://arxiv.org/abs/2410.20436
- SAMPro3D：https://arxiv.org/abs/2311.17707
- AGILE3D：https://ywyue.github.io/AGILE3D
- Point-SAM：https://point-sam.github.io
- NOAA SfM 群体生长率数据：https://catalog.data.gov/dataset/structure-from-motion-sfm-coral-vital-rates-annotations-derived-from-fixed-site-imagery-ac-20143
- Zenodo 珊瑚点云数据：https://zenodo.org/records/14249838
- DeepReefMap 示例数据：https://zenodo.org/record/10624794
- PICOGRAM（2026，2D 单株分割，平均 1.8 次点击达到 IoU 0.90）：https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2026.1716796/full
