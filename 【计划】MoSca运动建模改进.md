# 计划：MoSca 运动建模改进

## 当前问题

MoSca 当前的动态建模流程大致是：

```text
动态 tracks
  -> 动态 3D curves
  -> MoSca scaffold nodes: node_xyz[t, m], node_rotation[t, m]
  -> 动态 GS 从动态 mask 区域采样
  -> 每个动态 GS 记录 ref_time 和 attach_ind
  -> 渲染目标帧 t 时，从 ref_time 通过 scaffold warp 到 t
```

这个设计已经有一定的多帧锚定基础，因为动态 GS 并不全都绑定到第 0 帧，而是各自带有 `ref_time`。但是当前主要问题是：

- 远离当前帧的动态 GS 也可能参与渲染，长距离 warp 容易带来漂移和重影。
- 动态 GS 的初始化来自所有时间帧，缺少明确的多规范帧 anchor 组织方式。
- scaffold 运动主要通过完整时间轨迹优化得到，缺少显式的双向预测一致性约束。
- 遮挡、快速运动、mask 不完整时，单一 ref_time 到目标帧的 warp 容易不稳定。

## 方向一：局部多规范帧激活

这是最容易落地的改动，对应现有代码里的 `nn_fusion` 机制。

当前 `DynSCFGaussian.forward(t)` 默认 `nn_fusion=-1`，即所有 `ref_time` 的动态 GS 都会被 warp 到目标帧 `t` 并参与渲染。可以改成只激活距离目标帧最近的若干个 `ref_time`：

```text
nn_fusion = 1：只使用最近的 1 个 ref_time
nn_fusion = 3：只使用最近的 3 个 ref_time
nn_fusion = 5：只使用最近的 5 个 ref_time
```

这样可以把每个 `ref_time` 显式解释为一个局部 canonical anchor：

```text
目标帧 t 只由邻近 canonical anchors 的动态 GS 负责渲染
远距离 anchor 的动态 GS 不参与当前帧，减少长距离形变误差
```

第一版实验建议：

```yaml
gs_dynamic_nn_fusion: 3
```

如果硬筛选造成 temporal flicker，可以进一步改成软权重：

```text
w_i(t) = exp(-|ref_time_i - t| / tau)
opacity_i(t) = opacity_i * w_i(t)
```

这可以写成：

```text
Multi-Canonical Temporal Anchor Fusion
```

## 方向二：Anchor-aware 动态 GS 初始化

这是更明确的多规范帧锚定机制。

当前动态 GS 从所有帧的 `dyn_mask * dep_mask` 中采样，每个 GS 以采样来源帧作为 `ref_time`。可以改成先选择一组规范帧 anchors，再围绕这些 anchors 初始化动态 GS。

建议设计：

```text
1. 每隔 K 帧选择一个 anchor frame，例如 K=20 或 K=30。
2. 也可以根据 SAM3 mask 质量选择 anchors，例如动态主体完整、遮挡少、运动模糊弱的帧。
3. 动态 GS 主要从 anchor frames 初始化。
4. 非 anchor 帧可以低比例采样，或者只用于优化监督，不直接生成大量独立动态 GS。
5. 每个动态 GS 绑定 anchor_id/ref_time。
6. 渲染目标帧时，只激活邻近 anchors 的动态 GS。
```

这样模型不再是“每帧都零散地产生动态 GS”，而是：

```text
多个局部 canonical frames 共同表示动态物体
每个 canonical anchor 负责一段时间窗口内的形变与渲染
```

第一版实验建议：

```yaml
gs_dynamic_anchor_stride: 20
gs_dynamic_anchor_sample_ratio: 1.0
gs_dynamic_non_anchor_sample_ratio: 0.2
gs_dynamic_nn_fusion: 3
```

预期收益：

- 动态 GS 组织更稳定。
- 减少远距离 ref_time 的错误贡献。
- 对遮挡和快速运动更稳。
- 更容易在论文/报告中表述为“多规范帧锚定机制”。

## 方向三：双向帧预测与双向形变场

可以做，而且建议作为第二阶段加入。

当前 MoSca 的 `scaffold.warp()` 已经支持从任意源时间 `src_t` 到目标时间 `dst_t` 的形变：

```text
W_{src -> dst}(x)
```

因此可以把运动估计从单向约束扩展为双向约束：

```text
forward:  t -> t + Δ
backward: t + Δ -> t
```

### 3.1 双向 cycle consistency

对动态 scaffold nodes 或动态 GS centers 加 cycle loss：

```text
x_t'  = W_{t -> t+Δ}(x_t)
x_t'' = W_{t+Δ -> t}(x_t')

L_cycle = || x_t'' - x_t ||
```

这个约束让 forward deformation 和 backward deformation 互相一致，减少单向 warp 的漂移。

建议优先作用在 scaffold nodes 上，而不是所有动态 GS 上：

```text
scaffold nodes 数量少，计算更稳
运动结构更接近 MoSca 的核心形变场
不容易被单个 GS 的颜色/opacity 优化干扰
```

### 3.2 双向 track reprojection loss

使用动态 track 监督前后两个方向：

```text
L_forward  = || project_{t+Δ}(W_{t -> t+Δ}(X_t)) - u_{t+Δ} ||
L_backward = || project_t(W_{t+Δ -> t}(X_{t+Δ})) - u_t ||
```

其中只使用满足条件的点：

```text
track 在两帧都 valid
点落在 SAM3 dynamic mask 内
深度有效
没有明显遮挡或越界
```

这样可以让物体运动不是只从单帧向外预测，而是由前后两帧共同约束。

### 3.3 Anchor-to-anchor 双向预测

如果使用多规范帧 anchors，可以进一步改成 anchor 间双向预测：

```text
anchor_i -> anchor_j
anchor_j -> anchor_i
```

对相邻 anchors 加：

```text
L_anchor_cycle = || W_{j -> i}(W_{i -> j}(x_i)) - x_i ||
```

这比所有帧两两做双向约束更便宜，也更符合多规范帧锚定机制。

## MoRel 启发：Anchor Relay 式双向融合

MoRel 的核心思想是：

```text
Global Canonical Anchor
  -> 周期性 Key-frame Anchors
  -> anchor 之间学习 forward/backward deformation
  -> 中间帧通过 bidirectional blending 平滑融合
```

它主要是在 deformation-field 基础上做 anchor relay 和 bidirectional blending。MoSca 不能直接照搬，因为 MoSca 的运动表示不是 MLP deformation field，而是：

```text
MoSca scaffold nodes + skinning + dynamic GS ref_time
```

所以更合理的迁移方式是：

```text
MoRel: anchor deformation field 双向融合
MoSca: anchor scaffold / dynamic GS 双向运动融合
```

### 4.1 将 `ref_time` 升级为 Key-frame Anchor

MoSca 当前每个 dynamic GS 已经保存 `ref_time`，表示它从哪一帧初始化。可以把 `ref_time` 从普通采样时间升级为显式的 key-frame anchor：

```text
anchor frames = {0, K, 2K, 3K, ...}
每个 dynamic GS 绑定一个 anchor_id/ref_time
每个 anchor 负责附近一段时间窗口
```

这相当于 MoSca 版本的 key-frame canonical anchor。和 MoRel 不同的是，这里 anchor 不需要重新训练一个独立 deformation field，而是使用已有 scaffold 的 `warp(src_t -> dst_t)`。

### 4.2 从单向预测改成左右 anchor 双向预测

对于目标帧 `t`，找到左右两个 anchor：

```text
left anchor:  a
right anchor: b
a <= t <= b
```

从左 anchor 做 forward prediction：

```text
X_t^f = W_{a -> t}(X_a)
```

从右 anchor 做 backward prediction：

```text
X_t^b = W_{b -> t}(X_b)
```

然后通过时间权重融合：

```text
w_a = (b - t) / (b - a)
w_b = (t - a) / (b - a)
```

理想形式：

```text
X_t = w_a * X_t^f + w_b * X_t^b
```

但第一版不建议直接混合 3D Gaussian 坐标，因为两个 anchor 的 warped 坐标、旋转、scale 可能不严格对应，直接混合可能破坏局部刚性。

更稳妥的第一版是 **render / opacity-level blending**：

```text
Render_t =
  w_a * Render(W_{a -> t}(G_a))
  +
  w_b * Render(W_{b -> t}(G_b))
```

在代码上可以近似为：

```text
属于 left anchor 的动态 GS：opacity *= w_a
属于 right anchor 的动态 GS：opacity *= w_b
其它远距离 anchor 的动态 GS：opacity *= 0
```

这就是 MoSca 版本的 intermediate frame bidirectional blending。

### 4.3 从 `nn_fusion` 硬筛选扩展为双向 opacity blending

当前 `nn_fusion` 是硬筛选：

```text
只保留最近 K 个 ref_time 的动态 GS
其它 GS opacity 直接为 0
```

可以改成：

```text
1. 找到目标帧 t 左右两个 anchor。
2. 只保留这两个 anchor 或附近 K 个 anchors。
3. 根据时间距离计算 opacity weight。
4. 对不同 anchor 的动态 GS 乘不同 opacity weight。
```

可配置项建议：

```yaml
gs_dynamic_anchor_stride: 20
gs_dynamic_anchor_blend: linear
gs_dynamic_anchor_blend_tau: 10.0
gs_dynamic_anchor_blend_k: 2
```

其中：

```text
linear：只用左右 anchor 线性融合
softmax：使用最近 K 个 anchor，按 exp(-|t-ref_time|/tau) 融合
hard：退化为原始 nn_fusion
```

### 4.4 双向 anchor cycle consistency

为了让左右 anchor 的运动预测彼此一致，可以加入 anchor 级 cycle loss。

对相邻 anchors `a` 和 `b`：

```text
X_b^pred = W_{a -> b}(X_a)
X_a^cycle = W_{b -> a}(X_b^pred)
```

约束：

```text
L_anchor_cycle = || X_a^cycle - X_a ||
```

同时可以做反向：

```text
X_a^pred = W_{b -> a}(X_b)
X_b^cycle = W_{a -> b}(X_a^pred)
L_anchor_cycle_rev = || X_b^cycle - X_b ||
```

最终：

```text
L_bidir_anchor =
  L_anchor_cycle
  +
  L_anchor_cycle_rev
```

这个 loss 最好先作用在 scaffold nodes 上，而不是直接作用在所有动态 GS 上。

### 4.5 与 SAM3 mask 的结合

SAM3 mask 可以用于选择更可靠的 anchors：

```text
动态主体面积适中
遮挡少
mask 连续性好
运动模糊弱
```

也可以用于过滤双向约束：

```text
只有 forward/backward 投影都落在 SAM3 dynamic mask 内的点，才参与双向 track reprojection loss。
```

这样可以减少背景点、遮挡点和错误 track 对双向运动场的干扰。

## 候选点子池与消融实验设计

接下来不应该把这些想法当成一条固定路线，而应该把它们当成候选模块逐个实验。每个模块都单独打开或关闭，观察指标和可视化是否提升。有效模块保留，无效模块回退。这个过程本身就是消融实验。

### 点子 A：`nn_fusion` 局部多规范帧激活

改动：

```text
保持当前动态 GS 初始化方式不变。
只在渲染时限制动态 GS 的时间来源。
目标帧 t 只使用最近 K 个 ref_time 的动态 GS。
```

配置候选：

```yaml
gs_dynamic_nn_fusion: -1  # baseline，所有 ref_time 都参与
gs_dynamic_nn_fusion: 1
gs_dynamic_nn_fusion: 3
gs_dynamic_nn_fusion: 5
```

预期收益：

```text
减少远距离 ref_time warp 带来的重影和漂移。
降低动态物体 temporal ghost。
```

主要风险：

```text
K 太小会导致动态 GS 覆盖不足。
anchor 切换处可能出现闪烁。
```

验证方式：

```text
先只跑 apple/block/paper-windmill。
比较 mPSNR/mSSIM/mLPIPS/PCK。
重点看动态物体边缘、遮挡恢复和 temporal flicker。
```

### 点子 B：Soft Anchor Opacity Blending

改动：

```text
把 nn_fusion 的硬筛选改成软权重。
动态 GS 的 opacity 根据 |ref_time - t| 衰减。
```

候选形式：

```text
linear：左右两个 anchor 线性融合。
softmax：最近 K 个 anchor 按 exp(-|ref_time-t|/tau) 融合。
gaussian：按 exp(-|ref_time-t|^2 / 2sigma^2) 融合。
```

配置候选：

```yaml
gs_dynamic_anchor_blend: linear
gs_dynamic_anchor_blend: softmax
gs_dynamic_anchor_blend_k: 2
gs_dynamic_anchor_blend_k: 3
gs_dynamic_anchor_blend_tau: 10.0
```

预期收益：

```text
比硬 nn_fusion 更平滑。
减少 anchor 切换导致的闪烁。
体现 MoRel 式 intermediate frame blending，但融合对象是 MoSca 的 dynamic GS contribution。
```

主要风险：

```text
过宽的融合窗口会重新引入远距离 ghost。
过窄的融合窗口可能仍有闪烁。
```

验证方式：

```text
与点子 A 的 hard nn_fusion 对比。
重点看视频播放时的 temporal consistency。
```

### 点子 C：Anchor-aware 动态 GS 初始化

改动：

```text
不再从所有帧平均采样动态 GS。
先选择 key-frame anchors。
动态 GS 主要从 anchor frames 初始化。
非 anchor 帧只低比例采样，或只参与监督。
```

配置候选：

```yaml
gs_dynamic_anchor_stride: 20
gs_dynamic_anchor_stride: 30
gs_dynamic_anchor_sample_ratio: 1.0
gs_dynamic_non_anchor_sample_ratio: 0.0
gs_dynamic_non_anchor_sample_ratio: 0.2
```

预期收益：

```text
动态 GS 分布更有组织。
多规范帧锚定更明确。
减少每帧零散初始化造成的冗余和不一致。
```

主要风险：

```text
anchor 选得不好会漏掉某些时间段的新出现区域。
non-anchor 采样太少可能导致动态物体覆盖不足。
```

验证方式：

```text
统计每个 ref_time 的动态 GS 数量。
观察非 anchor 帧是否仍能完整渲染动态物体。
比较存储量和渲染质量。
```

### 点子 D：SAM3-guided Anchor Selection

改动：

```text
anchor 不只按固定 stride 选择。
结合 SAM3 mask 质量选择更可靠的 anchor frames。
```

候选评分：

```text
mask_area：动态区域面积适中。
mask_stability：相邻帧 mask 面积变化平滑。
sharpness：图像运动模糊较低。
visibility：动态主体无遮挡比例高。
```

预期收益：

```text
anchor 更可能落在物体清晰、可见、分割完整的帧。
提升 anchor 初始化质量。
```

主要风险：

```text
评分规则可能偏向静止或运动较小的帧。
如果 SAM3 mask 本身错误，会选择错误 anchor。
```

验证方式：

```text
对比 fixed-stride anchors 与 SAM3-selected anchors。
先只做 anchor 可视化和动态 GS 初始化质量检查，再跑完整训练。
```

### 点子 E：左右 Anchor 双向 Render Blending

改动：

```text
目标帧 t 不只由一个 ref_time 预测。
从左 anchor forward warp 到 t。
从右 anchor backward warp 到 t。
最后融合两侧 anchor 的动态 GS 渲染贡献。
```

第一版建议只融合 opacity/render contribution：

```text
left anchor GS:  opacity *= w_left(t)
right anchor GS: opacity *= w_right(t)
其它 anchors:   opacity *= 0
```

暂时不直接融合 3D 坐标：

```text
不做 X_t = w_left * X_forward + w_right * X_backward
```

预期收益：

```text
让目标帧由前后两个规范帧共同解释。
减少单侧预测在遮挡、快速运动处的错误。
更明确体现“双向帧预测”。
```

主要风险：

```text
左右 anchor 的 GS 不一定一一对应，render-level blending 可能出现双影。
如果左右 anchor 都不准，会叠加错误。
```

验证方式：

```text
先和点子 B 对比。
重点看 anchor 中间帧是否更稳定。
检查是否出现双影。
```

### 点子 F：Scaffold Node 双向 Cycle Loss

改动：

```text
在 geometry_scf_init 阶段加入 node-level 双向循环一致性。
```

形式：

```text
x_t'  = W_{t -> t+Δ}(x_t)
x_t'' = W_{t+Δ -> t}(x_t')
L_cycle = || x_t'' - x_t ||
```

配置候选：

```yaml
geo_bidir_cycle_weight: 0.01
geo_bidir_cycle_weight: 0.05
geo_bidir_cycle_deltas: [1, 3, 6]
```

预期收益：

```text
约束 scaffold 运动场前后方向一致。
减少单向运动漂移。
对快速运动和长时序更有帮助。
```

主要风险：

```text
如果相机或深度不准，cycle loss 可能强化错误形变。
loss 太大可能过度平滑动态运动。
```

验证方式：

```text
先只加到 scaffold nodes，不加到所有 dynamic GS。
观察 PCK 是否提升，以及动态物体是否过度平滑。
```

### 点子 G：双向 Track Reprojection Loss

改动：

```text
使用动态 tracks 约束 forward 和 backward 形变。
```

形式：

```text
L_forward  = || project_{t+Δ}(W_{t -> t+Δ}(X_t)) - u_{t+Δ} ||
L_backward = || project_t(W_{t+Δ -> t}(X_{t+Δ})) - u_t ||
```

只使用可靠点：

```text
两帧 track 都 valid。
点落在 SAM3 dynamic mask 内。
深度有效。
投影不越界。
```

预期收益：

```text
直接提升动态运动轨迹的一致性。
更贴近 PCK 指标。
```

主要风险：

```text
track 噪声会直接影响形变场。
实现复杂度高于 node cycle loss。
```

验证方式：

```text
先只在少量 sampled tracks 上计算。
与点子 F 对比，看 PCK 是否明显提升。
```

### 点子 H：Feature/Residual-guided Dynamic Densification

改动：

```text
借鉴 MoRel 的 feature-variance-guided densification。
在 MoSca 中用渲染残差、SAM3 mask 边界、动态 track error 来决定动态 GS 或 scaffold node 的 densification。
```

候选信号：

```text
render residual 高。
SAM3 mask 边界附近。
track reprojection error 高。
动态 GS gradient 高。
```

预期收益：

```text
把容量分配到动态主体细节、遮挡边界和高频区域。
减少无效背景动态 GS。
```

主要风险：

```text
如果 residual 来自相机错误或 mask 错误，会错误增密。
可能增加训练时间和显存。
```

验证方式：

```text
统计动态 GS/node 数量增长。
观察边界细节是否提升。
比较存储量和指标变化。
```

## 消融实验矩阵

建议每次只打开一个新模块。基础 baseline 使用当前最佳 MoSca 配置，例如：

```text
SAM3 mask replace
colfree 或 gtcam 固定一种
track identification 固定一种
nn_fusion=-1
无 anchor blending
无 bidirectional loss
```

第一组消融：

```text
Baseline
+ A: hard nn_fusion=3
+ A: hard nn_fusion=5
+ B: soft anchor opacity blending
+ C: anchor-aware dynamic GS initialization
```

第二组消融：

```text
Best from first group
+ D: SAM3-guided anchor selection
+ E: left/right anchor render blending
+ F: node-level bidirectional cycle loss
+ G: bidirectional track reprojection loss
```

第三组组合实验：

```text
Best A/B/C
Best A/B/C + E
Best A/B/C + F
Best A/B/C + E + F
Best A/B/C + E + F + G
```

保留标准：

```text
mPSNR / mSSIM 提升，mLPIPS 下降。
PCK 不明显下降，最好提升。
动态区域 ghost 减少。
视频播放 temporal flicker 减少。
存储量和耗时可接受。
```

如果某个模块只提升定性但降低指标，需要单独保存结果，用于后续分析。

## 推荐落地顺序

### 第一阶段：低风险验证

只改动态 GS 的时间激活：

```text
开启 nn_fusion=3 或 nn_fusion=5
保持原始 scaffold 优化和 photometric loss 不变
观察动态区域重影、mPSNR/mSSIM/mLPIPS/PCK 是否改善
```

### 第二阶段：明确多规范帧锚定

加入 anchor-aware dynamic GS initialization：

```text
选择 anchor frames
动态 GS 主要从 anchor frames 初始化
非 anchor 帧减少采样或不采样
渲染时只使用邻近 anchors 的动态 GS
```

### 第三阶段：Anchor Relay 式双向融合

把 `nn_fusion` 从硬筛选改成左右 anchor 的 opacity blending：

```text
left anchor forward warp 到目标帧
right anchor backward warp 到目标帧
根据时间距离融合两侧 anchor 的动态 GS 贡献
```

第一版只做 opacity/render contribution blending，不直接混合 3D 坐标。

### 第四阶段：双向形变约束

在 scaffold 优化阶段加入：

```text
node-level bidirectional cycle loss
dynamic-track bidirectional reprojection loss
anchor-to-anchor cycle loss
```

优先级建议：

```text
1. nn_fusion 局部 anchor 激活
2. anchor-aware 动态 GS 初始化
3. 左右 anchor opacity blending
4. node-level bidirectional cycle loss
5. dynamic-track bidirectional reprojection loss
```

## 风险

- `nn_fusion` 太小可能导致某些帧动态 GS 覆盖不足。
- anchor 间切换如果是硬切换，可能造成闪烁，需要软权重平滑。
- 直接混合左右 anchor 预测出的 3D 坐标可能破坏 Gaussian 局部刚性，第一版应优先做 opacity/render-level blending。
- 双向约束依赖 camera/depth/track 质量，colfree 场景下相机不准可能会放大错误。
- SAM3 mask 如果漏掉运动部件，anchor 初始化会缺少对应动态 GS。

## 判断标准

需要同时看定量和定性：

```text
mPSNR / mSSIM / mLPIPS
PCK@0.05
动态主体边界是否更干净
是否减少重影和漂移
anchor 切换是否闪烁
新视角下动态物体是否更完整
```
