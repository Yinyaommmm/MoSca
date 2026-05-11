# 计划：使用 SAM3 引导 DyCheck 动态掩码改进 MoSca

## 动机

MoSca 当前主要依赖极几何不一致性和 track 级运动线索来区分静态区域与动态区域。这种方式适合发现运动，但由它生成的像素级动态掩码可能比较噪声：

- 地面等静态背景可能因为深度、光流、相机位姿或遮挡误差被误判为动态区域。
- 运动物体可能因为 track 覆盖稀疏、像素级 mask 从动态曲线最近邻扩展而出现残缺。
- 细长结构、遮挡区域、重新出现的物体部分尤其容易不稳定。

DyCheck 场景里的动态主体通常比较明确，因此适合使用 prompt-based 视频分割。SAM3 可以用来生成更干净、更完整的运动主体物体级掩码。

## 总体方向

SAM3 应该作为物体边界补全和时序传播模块使用，而不是作为唯一的运动判断来源。

计划流程如下：

1. 对每个 DyCheck sequence 选择一个或多个关键帧。
2. 在关键帧上对动态主体提供 prompt。
3. 使用 SAM3 视频分割，将动态主体掩码传播到整个视频序列。
4. 将传播后的掩码保存到对应 sequence 下，例如：

   ```text
   <sequence>/
     dynamic_mask_sam3/
       00000.png
       00001.png
       ...
   ```

5. 修改 MoSca 的数据读取或重建代码，使 `dynamic_mask_sam3` 可以替换或修正默认的 `s2d.dyn_mask`。

## 接入方案

### 方案 A：直接替换像素级动态掩码

在 Gaussian 初始化阶段，直接将 SAM3 掩码作为 `s2d.dyn_mask`：

- 静态 GS 初始化使用 `~sam3_mask * dep_mask`。
- 动态 GS 初始化使用 `sam3_mask * dep_mask`。

这个方案最简单，也最可能改善物体边界，但效果高度依赖 prompt 质量。

### 方案 B：SAM3 掩码结合 EPI 过滤

将 SAM3 掩码与 MoSca 的 EPI 动态掩码结合：

- 保守模式：`dynamic = sam3_mask & epi_dyn_mask`
- 完整模式：`dynamic = sam3_mask | epi_dyn_mask`
- 首轮实验建议：以 `sam3_mask` 作为主掩码，同时保留 EPI 掩码用于可视化和诊断。

这个方案既保留了运动几何线索，也允许 SAM3 提供更完整的物体边界。

### 方案 C：作为软辅助监督

不直接硬替换 `s2d.dyn_mask`，而是保留 MoSca 原始动态掩码，同时增加一个 soft consistency loss，鼓励渲染出的动静分离结果与 SAM3 掩码一致。

这个方案训练风险更低，但需要更多代码改动。

## 预期收益

- 动态主体掩码更完整。
- 地面和背景泄漏到动态区域的情况减少。
- 动态 Gaussian 初始化更干净。
- 对动态区域的新视角合成质量可能提升，尤其是前景主体明确的场景。

## 风险

- SAM3 分割的是物体，不是运动。若 prompt 到静态物体，它也会被当成动态。
- prompt 不完整时，可能漏掉次要运动物体、手持物或被带动的物体。
- 如果错误 mask 被当成硬 GT，会直接伤害重建质量。
- 遮挡和重新出现仍需要人工检查或时序后处理。

## 第一轮实验

先在一小部分 DyCheck 场景上生成 `dynamic_mask_sam3`：

- `apple`
- `block`
- `paper-windmill`
- `space-out`

然后给 MoSca 增加一个最小代码路径，使其可以选择：

```bash
dynamic_mask_mode=sam3
```

与当前 `origin` 结果对比以下指标：

- `mPSNR`
- `mSSIM`
- `mLPIPS`
- `PCK@0.05`
- 定性渲染视频

第一版实现不应该修改相机估计流程，只影响 Gaussian 初始化和 photometric reconstruction 阶段使用的静态/动态像素掩码。
