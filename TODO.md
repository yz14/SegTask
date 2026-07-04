# 规则与要求

## 一、任务启动(两阶段,强制)
新任务必须**先调研规划、后编码执行**,二者分属不同轮次,不得在同一轮合并完成。

- **第一轮(调研规划,禁止写实现代码)**:阅读相关现有代码与上下文,检索业界做法;明确目标、范围边界、约束与难点;产出一份**可拆分、各步骤可独立执行**的计划,每步含目标、预期产出、验收标准及依赖关系。本轮只输出结论与计划,等我确认后再动手。
- **第二轮起(执行)**:计划确认后才开始编码,严格按步推进,每轮专注一步并保证质量。计划需调整时先说明原因,不擅自偏离。

## 二、质量第一
- 充分思考分析后再动手,宁可慢不可糙。
- **单轮质量优先于单轮完整性**:宁可这轮少做做透,不贪多做糙。

## 三、范围克制
- 不擅自修改任务范围外的代码、不做无关重构。确需引入新的第三方依赖时,先说明用途和必要性再使用(避免为小功能引入重型库或重复造轮子)。

## 四、代码质量
- 模块化设计,职责分离,不把所有代码堆在一个文件。
- 复用优先,避免重复代码。
- 涉及具体库/API 时以最新官方文档为准,不凭记忆臆断版本与用法。

## 五、调试支持
- 需要时加入日志辅助定位:正式日志走 logging 并保留;临时调试代码统一加 `# DEBUG` 标记,便于稳定后清理。

## 六、完成与自查
- 完成后自查:是否真正达成目标;改动是否破坏了原本正常的功能(尤其与改动处有依赖关系的地方);边界情况是否处理。并说明如何验证。

## 七、沟通规范
- **开始前**:复述你理解的目标与将遵守的规则。
- **进行中**:如需拆分,明确告知本轮要完成什么。
- **完成后**:总结本轮成果与后续计划。


本地测试环境为: **D:\miniconda\envs\torch27_env\python.exe**。  


segtask_v1是2.5D/3D分割项目(见对应README.md)
gentask是生成/超分项目。大多数设计和代码和分割代码非常类似，注意：  
生成/超分不一定有img-lable对；可能只有lable(高质图像)，img要对应任务的退化。可能有bbox, region_weight。
任务可能是：  
厚层生成薄层(d,H,W -> D,H,W)，可能需要逼真的薄层模拟出厚层（例如部分容积效应等等）。  
面内超分(D,h,w -> D,H,W)。  
ssltask是自监督学习项目，主要是对分割，生成，分类（暂缺），检测（暂缺）的backbone（部分/全部）进行预训练，提升下游模型指标。


# TODO  
1 优化gentask里面的代码，目前主要用于超分，先针对CT超分进行修改、适配和优化（如需用到分割项目的代码，可先拷贝py文件后修改）：  
image_dir: 低分辨率输入可能缺失，label_dir: 高分辨率GT，bbox_dir : 可能有，region_weight_dir : 可能有，npz_dir: 制作好的路径  

a 厚层生成薄层  
GT是薄层，对于的厚层可能缺失，需要针对性的用数据增强算法从薄层模拟出厚层（一张厚层切片是层厚范围内体素的平均(部分容积效应)）  

输入可以是对厚层的z轴滑块，例如取连续两张切片作为起始和终点，然后模型往中间插切片2x,4x，类似这样，模型对应的可能需要采用VFI；输入还可以是在厚层的x或者y进行滑块，由于只有z轴是低分辨率，所以模型对应的可能采用SISR只对z轴2x,4x，类似这样超分。  

输入可以输入更多FOV来帮助模型，例如薄层是1,2,3,4,5,6,7切片，厚层是1,3,5,7，对z轴滑块的基本输入是，例如3,5(1x FOV)->3,4,5，多FOV就是3,5(1x)+1,3,5,7(2x)->3,4,5；如果是在x轴取切片，那么输入就和分割的2.5D方案非常类似，例如C(x轴),H(y),w(z)->C,H,W，在x轴上取更多的切片，例如Cx1.5，Cx2个切片作为多个FOV  

数据读取部分几乎可以沿用分割任务中2.5D方案，例如保留oversample这样数据增强后边界的异常可以通用中心剪裁去掉；只在数据增强结束后才做成2D数据等等；我理解的就是把输入图像换成低分辨率，标签换成高分辨率，区域权重可能是重要的器官等等；只对oversample的max FOV处理，然后中心剪裁出多FOV等等。    

模型对应的用自然图像公认的经典、关键、高质量超分算法（SISR/VFI）；安装分割的代码风格来写(虽然不能统一UNet框架，但是构造模型的过程的block, stage, encoder, decoder这些可以参考)。  

输出对应的就是插帧/超分。  

需要针对性的适配各种技巧（例如在分割中有很多的技巧，例如EMA，deep supervision, 多个输出头来辅助监督等等）

b 面内超分（x,y是低分辨率，D,h,w -> D,H,W）
这个更加直观，也更加容易在分割的2.5D方案上适配(类似a中的在x轴取切片，然后超分z轴，只不过现在是在z轴取切片然后需要同时超分x,y轴)，和自然图像任务最接近。


2 分割项目代码分析审查，需要认真，详细的分析，思考和调研。保证高质量完成。为了保证高质量完成，本轮不动任何代码/文档：  
本轮关注模型/算法架构的方向，例如unet是否有升级的地方，注意力是否有更多更好的算法，这些可能需要细致的调研现有的高质量论文/项目来借鉴（公认高质量自然图像，NLP，LLM论文，项目算法）。我说一下我之前怎么做的，先实现了unet，升级到unet++，升级加入multirf, qkv attn, linear attn等等这些。我希望你也类似朝算法方向给我一些高质量模型/算法提质的建议。  

有没有加速训练，减少训练/推理GPU占用高质量技巧，目前我只实现了grad_checkpointing, 多卡训练，混合精度训练等等。  

进展：  

A1 — SDPA 替换 ✅ 正确实施
@d:/codes/work-projects/SegTask/segtask_v1/models/blocks.py:658-685 的 _SoftmaxQKVAttention 已用 F.scaled_dot_product_attention,手写 einsum + fp32 softmax 已删除;QKV/proj 仍为 Conv1d,权重兼容。有等价性测试(但测试本身有 bug,见下)。

小瑕疵:

@d:/codes/work-projects/SegTask/segtask_v1/models/blocks.py:673 的 ch = width // (3 * self.num_heads) 是死变量;
@d:/codes/work-projects/SegTask/segtask_v1/models/blocks.py:643-655 的 _sdpa_from_qkv 全库无任何调用点,是死代码,建议删除。

A2 — RoPE + GEGLU FFN ✅ 正确实施
RoPE(_apply_rope_nd,blocks.py:406-453):按轴分配 head_dim、无参数、支持 position_offsets;数学性质有测试保证(模长保持、相对 logits 平移不变)。
FFN(blocks.py:860-887):GEGLU + zero-init 输出投影,初始恒等;ffn_norm 独立 PreNorm,符合标准 Transformer 块。
守卫正确:use_rope 对 linear/grid 显式报错(blocks.py:823-832)。
配置贯通(config.py:423-426 → factory.py:130-132)、state_dict 键集合测试齐全。

A3 — Window/Grid 注意力 ✅ 正确实施,但有一处无效计算
partition/unpartition、padding mask(finfo.min 加性 mask)、非整除尺寸裁回均正确;locality 测试(窗口内局部、grid 跨窗)验证了语义。
配置逐 stage 可指定 'window'|'grid',softmax O(N²) 护栏对 window/grid 豁免(config.py:1391-1400 仅对 softmax 生效)——设计正确。
问题(性能级):@d:/codes/work-projects/SegTask/segtask_v1/models/blocks.py:717-730 window+RoPE 时按窗口逐个做 Python 循环施加带 offset 的 RoPE。但你们自己的测试 test_window_rope_relative_logits_with_offsets 恰好证明了:同窗口内 q/k 共享同一 offset 时,attention logits 与 offset 无关(RoPE 相对性)。即整个 per-window 循环在数学上等价于一次 offset=0 的全批量调用——浅层窗口数多时(如 16³ 特征 4³ 窗口 = 64 窗)这是 64 次串行 Python 调用的纯浪费。建议直接删循环、统一 offset=0。

当前实现每个 stage 只挂一个 SelfAttentionBlock,若只配 window 而不在别处配 grid,跨窗信息只能靠卷积传播——建议文档里点明"window 应与 grid 成对使用"。

A4 — UpKern + 膨胀重参数 ✅ 实施正确,但有 3 个实际缺陷
数值等价性(训练态 vs deploy 态 allclose、dilated kernel scatter 展开、端到端 build_model)测试齐全。但:

设备 bug(会真实咬人):@d:/codes/work-projects/SegTask/segtask_v1/models/mednext.py:227-232 switch_to_deploy 里 Conv(...) 未指定 device/dtype,在 CPU 上创建;GPU 模型调用 reparameterize_model 后 reparam 留在 CPU,下一次 GPU forward 直接报设备不匹配。测试仅覆盖 CPU 所以没暴露。最小修复:创建时传 device=weight.device, dtype=weight.dtype(或建后 .to())。
reparameterize_model 无调用点:全库只有定义,predictor/推理入口未接线。deploy 收益目前只能靠用户手工调用,建议在推理加载路径加开关。
checkpoint 兼容断裂未提示:dilated_reparam=True 的键是 dwconv.lk.weight/dwconv.branches.*,plain 训练的 checkpoint 是 dwconv.weight;pretrain_upkern 迁移时这类键会静默丢弃(upkern_remap_state_dict 只按同名匹配),用户从 k=3 plain checkpoint UpKern 到 k=5 reparam 模型时深度卷积全部保持随机初始化,仅有 missing-keys warning 可循。建议在 UpKern 路径显式检测并告警这种组合。
另两个备注级:upkern_remap_state_dict(mednext.py:286-289)未检查 shape[1]==1,非 depthwise 的同通道 conv 也会被插值(文档声称仅 depthwise);align_corners=True 与 MedNeXt 官方实现(默认 False)不一致——对 3→5 小核影响很小,但消融时应知情。

A5 — drop-path 通用化 / GRN / AttentionGate norm ✅ 全部实施
drop-path 已通用到全部 ResNet 变体(basic/preact/bottleneck/r2plus1d/MultiRF)+ MedNeXt,factory 统一 _make_drop_path_rates 线性递增,rates 按全局 block 序切片——正确。
GRN(blocks.py:74-92)符合 ConvNeXt-V2:空间 L2 → 通道均值除法 → 零初始化 gamma/beta,ConvNeXt/MedNeXt 均接了 grn_enabled。
AttentionGate(blocks.py:891-914)改为 get_norm(norm_type,...),默认 "batch" 保持兼容,三种 decoder 均传入 attn_gate_norm——完全符合计划。
一处不一致(设计级):ResNet 系所有块的 drop_path 都被 in_ch == out_ch 门控(如 resnet.py:43-45),即每个 stage 的首块(升通道)永远不生效 drop-path,其分配到的线性递增率被静默丢弃。而 ConvNeXt/MedNeXt 走 AdaptBlock(先 1×1 投影再进块),首块照常有 drop-path。数学上 shortcut 为投影时丢弃 residual 分支同样成立(res + 0 仍是有效路径),该门控过于保守且两族 backbone 行为不一致。建议统一(去掉门控)或在注释里说明理由。

B1–B3 / P1–P3 交叉核验
项	状态	证据
B1 no_sync()	✅ 已修	trainer.py:717-720,forward 也在 no_sync 内
B2 TF32	✅ 已修	utils.py:363-370,deterministic 时正确关闭
B2 fused AdamW	✅ 已修	optim.py:43-45,带 CUDA 门控
B2 EMA foreach	✅ 已修	utils.py:92-93 _foreach_mul_/_foreach_add_,整型 buffer 单独处理
B3 选择性 checkpoint	✅ 已修	grad_ckpt_encoder_stages 掩码 + 测试
P1 one_cycle warmup	✅ 已修	optim.py:92-101 映射 pct_start,trainer.py:153-155 关外层 warmup,raise 已删
P2 resume RNG 同流	✅ 已修	trainer.py:1030-1033 rank>0 重播种
P3 梯度非有限守卫	✅ 已修	trainer.py:776-785,且 skip 时 scheduler/EMA 照常推进(正确)
P4 回归测试	✅ 基本补齐	test_todo_p_regressions.py(keep-last-k、aspect 校正、val 确定性等)+ test_round2_fixes.py:337 compile+EMA 冒烟
二、测试验证结果
运行 pytest tests/test_selfattn.py test_a5_blocks.py test_dilated_reparam.py test_upkern.py test_mednext.py test_todo_p_regressions.py:108 通过,1 失败。

失败项 test_softmax_qkv_attention_matches_sdpa_reference 是测试自身的 bug,非产品代码问题:



test_selfattn.py:68
ref = ref.view(qkv.shape[0], h * c, qkv.shape[2])
torch.einsum 输出非连续,.view 抛 RuntimeError,应改为 .reshape。这说明该测试在当前环境下从未真正跑通过——SDPA 与旧手写实现的数值等价性实际上没有被回归覆盖住,建议一并修复(一行改动)。

三、全库剩余问题与优化空间(按优先级)
高(正确性/会咬人)

switch_to_deploy CPU 设备 bug(见 A4-1);
test_selfattn.py:68 .view→.reshape,恢复 SDPA 等价性回归;
reparam 与 plain checkpoint 键不兼容的静默丢弃(A4-3)。
中(性能/一致性) 4. window RoPE per-window 循环删除(数学等价,浅层可省大量 Python 开销); 5. RoPE 的 meshgrid + inv_freq + cos/sin 每次 forward 全量重算、无缓存(blocks.py:429-445)——固定 patch 尺寸下形状恒定,建议按 (spatial_shape, rot_dim) 缓存 cos/sin buffer; 6. ResNet 系 drop-path 的 in_ch==out_ch 门控与 ConvNeXt/MedNeXt 行为不一致(A5); 7. DilatedReparamBlock 内部 BN 在 3D batch=2 下的统计噪声——与你们批评 AttentionGate BN 的理由相同(UniRepLKNet 的 BN 面向大 batch ImageNet);fold 数学确实需要 BN,但训练期噪声风险应在消融中确认或注释点明。

低(卫生) 8. 死代码:_sdpa_from_qkv、_SoftmaxQKVAttention.forward 的 ch; 9. upkern_remap_state_dict 补 shape[1]==1 检查、align_corners 与官方对齐说明; 10. reparameterize_model 接入推理入口(带配置开关)。

仍在待办池(与 TODO 决定一致,非遗漏):B4 regional compile 实用化、B5(ZeRO/channels_last benchmark)、A6 Mamba、A7 clDice + largest-CC(P7)。