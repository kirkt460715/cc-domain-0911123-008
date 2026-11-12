# 试验数据版本库

以内容摘要保存试验数据，按 schema 校验和版本化派生，支持独立批准的数据发布及撤回管理；在此基础上提供可保存、可验证、绑定已发布版本的寿命实验分析。

Python 3.11 标准库与 SQLite 即可运行。数据、时钟和文件均在本地，不需要服务凭据。示例只使用合成记录。

## 运行

在仓库根目录执行：

```sh
python -m trialvault demo
python -m examples.context
python -m unittest discover -s tests -v
python -m compileall -q trialvault examples
```

示例使用内存或临时目录，不会改动持久台账。`examples/context.json` 记录一组可重复导入的业务输入，字段与时间口径见 `examples/数据说明.md`。`examples.context` 检查数据并调用现有产品入口；它只展示原有业务数据，不计算分析结论。

`tests` 覆盖现有业务生命周期、非法输入、版本和完整性边界，另有示例导入回归与寿命实验似然/删失/发布绑定边界。完整测试和上述演示可在一次短终端会话中执行。

## 寿命实验

`trialvault.life` 把一个已批准发布中的**固定数据集版本**做成一可保存的寿命实验。调用入口为 `life_run` / `life_get` / `life_list` / `life_export` / `life_verify`（`python -m trialvault run request.json`）。

### 输入

两种输入，同一份 schema 都按强类型校验：

* `inspection_long`（默认）：每行一次巡检，字段 specimen、batch、temperature_c、entry_day、inspection_day、result（`normal`/`failed`）；可选 `exact` 标记「该次巡检就是精确失效时刻」，可选 `removed`。
* `interval_summary`：每试件一行摘要，`left_day`（最后一次已知正常）、`right_day`（首次已知失效，仍运行则为空）、`kind`。配置项 `input_format` 与 `columns` 指定列名；`kind_choices` 可显式映射业务标签，未知标签直接报 `invalid_result`，绝不默认为删失。

所有温度输入是**摄氏**。Kelvin 按 SI 定义 **T_K = T_C + 273.15** 精确换算，任何地方都不用 273 近似。参考温度 `reference_temperature_c` 默认 25 °C；Arrhenius 用 z = 1/T_K − 1/T_ref_K，因而不随参考温度选择而改变形状。

### 删失口径

迟进入试件的暴露量从各自 entry_day 起算。设 τ = right − entry、λ = left − entry：

* 精确失效：`f(τ)`
* 区间删失：`S(λ) − S(τ)`（精确的生存函数增量，**不用区间中点**；用 `S(λ)·(1−e^{logS(τ)−logS(λ)})` 保持紧巡检区间精度）
* 右删失：`S(λ)`
* 延迟进入：上述时间一律是相对 entry 的暴露量

发现失效后又恢复、时间倒序、同一日重复巡检、同一试件元数据冲突、摄氏温度无效（≤ −273.15 °C 或超过配置上限）、左界晚于右界、entry 之后无有效巡检——全部在拟合前给出诊断；受影响试件逐行记录去留（`decisions`，含行号与原因），从不静默改边界或把首检失败当成精确失效。

### 模型与输出

比较两个 Weibull 模型：

* `common_scale_weibull`：尺度 η = exp(log_scale) 不随温度变化；
* `arrhenius_weibull`：η(T) = exp(log_scale + slope·z)，slope = Ea/kB（单位 K），报告 Ea（eV，1 eV = 8.617333262e-5·kB… 即 slope × 8.617333262e-5）。log_scale 是参考温度下的 log η。

以**批次隔离**的验证似然选择模型：批次按种子确定性划分（也可 `validation_batches` 指定），同批次绝不跨训练/验证；所有温度范围检查只看训练批次，外推用温度只依据训练温度范围。验证折不可拟合、缺少失效或参数不可识别时，状态降级并写明限制。

输出包括参数、优化收敛信息（迭代数、评估数、梯度范数、停止原因、多起点尝试）、指定使用温度的生存曲线、B10/B50/B90，及**按材料批次整体重采样**的 bootstrap 区间。重采样失败的次数、比例和错误码始终报告；失败比例过高或全失败会限制结论。

### 版本与交付

保存实验时冻结：数据集版本摘要与按内容寻址的 artifact、包含该版本的已批准发布（reviewer、manifest 摘要）、配置、种子、逐行去留和完整分析。取回时**不重算**，也**不悄悄改用最新版本**。若原发布被撤回/内容变化、绑定版本被撤回或 artifact 校验失败，结果照原样返回但标记 `suitable_for_delivery: false` 及原因，且拒绝导出。

`life_export` 生成含输入 artifact、`analysis.json`、`decisions.jsonl`、`observations.jsonl` 与签名 manifest 的目录；`life_verify --recompute` 从包内输入和配置重算并比对摘要，可独立核对完整性、输入血缘与可复现性。

`python -m trialvault demo` 的 `life_experiment` 段展示：区间失效与仍在运行试件、批次 bootstrap 区间、导出与重算验证、以及撤回发布后结果不再适合交付；`likelihood_check` 段给出可用手算核对的小样本似然值与温度单位换算。

## 容器

```sh
docker build -t local/trialvault .
docker run --rm --network none local/trialvault
docker run --rm --network none local/trialvault python -m examples.context
```

镜像构建完成后运行和测试不下载文件。持久化使用各自命令的本地路径参数；具体操作及参数可通过 `python -m trialvault --help` 查询。第三方依赖清单为空。
