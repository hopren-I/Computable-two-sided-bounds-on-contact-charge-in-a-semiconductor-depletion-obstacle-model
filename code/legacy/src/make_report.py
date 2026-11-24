"""Generate a data-driven report only after all numerical checks pass."""
from __future__ import annotations
import csv
import json
import math
from .paths import TABLES, REPORT
from .report_priority import report as priority_report


def read_csv(name):
    with (TABLES/name).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def fmt(x):
    if isinstance(x,bool):return '是' if x else '否'
    try:v=float(x)
    except (ValueError,TypeError):return str(x).replace('|','/')
    if not math.isfinite(v):return '不适用'
    if v==0:return '0'
    if v==int(v) and abs(v)<10000:return str(int(v))
    return f'{v:.4e}' if abs(v)<1e-3 or abs(v)>=1e4 else f'{v:.6f}'


def table(rows,cols,headers=None):
    heads=headers or cols
    return '\n'.join(['| '+' | '.join(heads)+' |','|'+'|'.join(['---']*len(cols))+'|']+
                     ['| '+' | '.join(fmt(r[k]) for k in cols)+' |' for r in rows])


def order(rows,key):
    return math.log(float(rows[-2][key])/float(rows[-1][key]))/math.log(float(rows[-2]['h'])/float(rows[-1]['h']))


def build_report():
    ver=json.loads((TABLES/'verification.json').read_text())
    if ver['n_checks']==0 or ver['n_checks']!=ver['n_passed']:
        raise RuntimeError('refusing to generate a success report from failed verification')
    manifest=json.loads((TABLES/'run_manifest.json').read_text())
    names=['m1_convergence','validation_1d','m2_charge_intervals','m2_capacitance_intervals',
           'm2_interface','m2_parameter_sweep','m2_nonmonotone','s2_branches','m5_poisson_boltzmann',
           'm3_full_contact','m3_partial_contact','m3_reference','m3_reference_bounds',
           'solver_crosscheck_2d','m3_reciprocity','m3_single_direction','s1_contact_width',
           'm4_hierarchy','m4_local_response','m4_updates']
    d={n:read_csv(n+'.csv') for n in names}
    m1=d['m1_convergence'];ref=d['m3_reference'];last=ref[-1]
    q=d['m2_charge_intervals'];best=min(q,key=lambda x:float(x['width']))
    intervals=d['m3_reference_bounds'];finest=[x for x in intervals if x['nx']==last['nx']]
    bestbound=min(finest,key=lambda x:float(x['width']))
    change=abs(float(last['Q_mixed'])-float(ref[-2]['Q_mixed']))
    eff=[float(x['effectivity_gap_over_energy_error']) for x in m1]
    maxkkt=max(float(x['stationarity_residual']) for x in ref)
    u=d['m4_updates']
    md=['# 半导体耗尽互补模型：修正后的数值实验报告','',
        f"运行时间（北京时间）：{manifest['started_at']}；Python {manifest['environment']['python']}，NumPy {manifest['environment']['numpy']}，SciPy {manifest['environment']['scipy']}。",
        '',f"本次完整运行通过 **{manifest['regression_tests']} 项回归测试**和 **{ver['n_passed']}/{ver['n_checks']} 项结果检查**。检查覆盖可行性、KKT、独立求积、能量包络、输出区间及更新恒等式。此计数代表所列有限测试的通过情况，不构成数学证明或区间算术认证。",
        '', '论文目录未改动，运行前后逐文件 SHA-256 一致。旧实验保存在 `backups/before-correction-20261020.tar.gz`。',
        '', '## 1. 本次修正与结论范围','',
        '- 活跃集在下一次组装前清零受约束节点；批量更新并在退出前验证 KKT，失败或迭代耗尽会报错。',
        '- 为避免高偏压细网格从全零场逐层释放节点，二维初始活跃集改由同边界无障碍 Poisson 解确定；后续迭代、离散模型和 KKT 验收不变。回归测试对 3 种接触、5 个偏压逐项比较该初值与全零初值所得解；独立优化器检查继续保留。',
        '- 一维电荷采用线性权函数的单元中点精确积分；二维直接电荷采用 Q1 边上的精确梯形积分。',
        '- 节点 KKT 残量除以全局装配的节点体积，再平均全部单元顶点密度。Dirichlet 反力不作为障碍乘子，边界密度由相邻内部节点延拓；不再用上限截断掩盖四倍归一化错误。这是一种非负重构，仍有重构误差。',
        '- 一维误差积分在非对齐界面处分段；界面和点态电容公式显式检查适用条件。PB 和电容真值不再固定为默认参数的 1。',
        '- 缺表、空表、非有限关键数据、失败检查均阻止结果发布；包含关系由数值端点重算。报告数值从本次 CSV 生成。',
        '', '## 2. M1：能量间隙、收敛与电荷恢复','',
        table(m1,['n','h','energy_error','gap','effectivity_gap_over_energy_error','L2_grad_error','Q_direct_error','Q_balance_error'],
              ['n','网格尺度','能量误差','间隙','间隙/能量误差','梯度L2误差','直接电荷误差','平衡电荷误差']),
        '',f"最后两层能量误差观察阶为 {order(m1,'energy_error'):.3f}，间隙观察阶为 {order(m1,'gap'):.3f}。在本序列中 effectivity 为 {min(eff):.3f}–{max(eff):.3f}；这是有限网格观测，不声称对所有偏压与网格一致有界。",
        '', '![M1收敛](../results/figures/m1_convergence.png)',
        '', '![M1电荷](../results/figures/m1_charge_order.png)',
        '', '补充 `validation_1d.csv` 覆盖 5 层网格与 10 个偏压，包括零偏压、旧求解器失败的小正偏压、非网格对齐界面、转折点及完全耗尽。直接验证 E≥E_exact≥D、梯度误差界、KKT 与间隙分解；没有加入边界噪声，因而不从这些实验声称抗噪优势。',
        '', '## 3. M2：界面与电荷/电容区间','',
        table(d['m2_interface'],['n','gap','delta','lower','upper','exact_interface']),
        '', '![界面](../results/figures/m2_interface.png)',
        '', '另构造保持非负与正确边界迹的非单调 P1 候选场，检查首次连续阈值穿越给出的包含区间。该测试展示定理不需要候选场单调，不能代替一般性证明。',
        '',table(d['m2_nonmonotone'],['positive_increments','gap','lower','upper','exact_interface']),
        '', '![非单调场](../results/figures/m2_nonmonotone.png)',
        '',table(q,['h','Q_minus','Q_plus','width']),
        '',f"扩展扫描实际包含内部最小值：所测步长中 h={best['h']} 的区间宽度最小，为 {fmt(best['width'])}。更小步长时能量界误差被除以 h 放大，更大步长时割线偏差增大。这里只报告所测离散步长中的最小值，不将其称为已求得的连续最优步长。",
        '',table(d['m2_capacitance_intervals'],['h','k','C_minus_average','C_plus_average','C_minus_point','C_plus_point','C_point_exact']),
        '', '![区间](../results/figures/m2_intervals.png)',
        '', '补充 `m2_parameter_sweep.csv` 覆盖 3 层网格、7 个偏压，分别检验电荷、平均电容及满足光滑条件时的点态电容。点态余项常数使用此解析算例可验证的 C″ 上界；一般器件未知该常数时，不能直接沿用点态区间。区间含真值和区间足够窄是两项不同要求。',
        '', '## 4. S2：部分与完全耗尽的转折','',
        table([x for x in d['s2_branches'] if 1.949<float(x['b'])<2.051],
              ['b','Q_exact','Q_balance','C_exact','C_numeric','right_derivative_exact','right_derivative_flux']),
        '', '![分支](../results/figures/s2_branches.png)',
        '', '用实际数值电荷和重构右端通量与解析值比较，并明确采样覆盖转折点两侧。完全耗尽段的人工 h/2 电荷偏差已消除。有限采样支持分支公式的一致性，连续性本身仍由解析公式给出。',
        '', '## 5. M3：二维结构验证与参考解','',
        'Q1 原始场、RT0 通量和 Q0 乘子使用一致积分。另以逐单元 Gauss 求积独立重算 E、D 和两项非负间隙分解；绝缘边界面自由度被剔除，平衡残差在所有单元检查。',
        '',table(d['m3_full_contact'],['nx','ny','Q_mixed','Q_direct','Q_reaction','gap','kappa']),
        '', '全接触例有解析电荷 1 和解析能量 1/3；能量上下界与解析值逐层比较。',
        '',table(ref,['nx','ny','Q_mixed','Q_direct','Q_reaction','gap','stationarity_residual']),
        '',f"局部接触参考网格为 {last['nx']}×{last['ny']}，混合电荷估计 {fmt(last['Q_mixed'])}。与前一层的变化为 {change:.6g}（相对 {100*change/abs(float(last['Q_mixed'])):.4g}%）；该变化不是对连续真误差的保证。上述参考序列的最大正自由节点驻值残差为 {maxkkt:.3e}。",
        '', 'Q_reaction 是离散最小能量对接触电势的导数，通过接触 Dirichlet 节点反力求和得到；它是另一种电荷评估途径，但仍属于同一 Q1 离散。Q_direct 是边界导数的精确边积分。三种计算随网格的变化同时报告，不仅比较两条恢复公式的一致性。',
        '', '![二维收敛](../results/figures/m3_2d.png)',
        '', '![二维参考与区间](../results/figures/m3_reference.png)',
        '', '### 5.1 连续模型电荷的能量割线区间','',
        table(intervals,['nx','h','Q_lower','Q_upper','width']),
        '',f"最细网格所测步长中的最窄数值区间为 [{fmt(bestbound['Q_lower'])}, {fmt(bestbound['Q_upper'])}]，宽度 {fmt(bestbound['width'])}。它由可行原始/对偶能量和凸性构成，独立于把最细网格当作真值的做法。浮点可行性与求积残差已经检查，但没有使用向外舍入，因此仍不称为机器严格认证。",
        '', '### 5.2 独立优化器交叉检查','',
        table(d['solver_crosscheck_2d'],['y0','y1','projected_gradient','max_u_difference','energy_difference']),
        '', 'L-BFGS-B 从独立线性初值求解同一离散 QP，与活跃集结果比较；覆盖全接触、中心局部接触和非对称接触。这检查求解器实现，不是独立网格离散或独立物理模型。',
        '', '### 5.3 互易与调和方向','',
        table(d['m3_reciprocity'],['metric','Q_sigma','Q_harmonic_identity','residual']),
        '',table(d['m3_single_direction'],['Q_before','Q_after','Q_target','gain','gain_formula','gain_residual']),
        '', '同一度量下的互易以及混用度量后的失配说明代数条件的重要性；单方向算例先沿调和方向扰动，再校正回来，用于核对增益公式。它们不单独证明真实器件电荷的精度，也不代表一般初值下的算法收敛。',
        '', '![场分布](../results/figures/m3_fields.png)',
        '', '![几何系数](../results/figures/m3_kappa.png)',
        '', '## 6. S1：接触宽度与电荷分解','',
        table(d['s1_contact_width'],['nx','contact_width','Q_mixed','Q_direct','kappa','geometric_term','screening_integral','gap']),
        '', '![接触扫描](../results/figures/s1_contact_width.png)',
        '', '在本次矩形、固定偏压与居中接触扫描中，Q 和 κ 均随接触宽度增加；仅对这组几何参数报告此趋势。模型为静电模型，这些结果不涉及电流输运。红色边界标出了局部接触，场图同时显示电势、重构乘子和离散调和权重。',
        '', '## 7. M4：可行场层次与实际局部更新','',
        table(d['m4_hierarchy'],['source_mesh','E','D','gap','Q','Q_difference_percent','reference_nx']),
        '', '该表是粗网格场映射到统一评价网格后的结果；电荷列报告相对更细参考值的差异，最后一行不再与自身比较。层次中的趋势是数值观察，不视为局部算法的收敛证明。',
        '', '![场层次](../results/figures/m4_hierarchy.png)',
        '', '### 7.1 全局响应与真实内部片区','',
        table(d['m4_local_response'],['max_abs_BR_plus_I','response_nonzero_fraction','patch_internal_faces','patch_boundary_flux','outside_divergence','patch_divergence_norm','patch_net_divergence']),
        '', '在真实 2×2 内部片区中，仅内部共享面赋非零通量，片区边界通量为零，延拓到全域后仍法向连续。片区内散度不恒为零，但积分净散度为零，片区外散度为零；这与先人为令单格散度等于零的旧测试不同。',
        '', '### 7.2 受约束乘子更新与原始场更新','',
        table(u,['iteration','E_before','E_after','D_before','D_after','gap_before','gap_after']),
        '',f"执行 {len(u)} 次局部更新：在所选 8 个单元求解 δ 的受约束二次问题，保持 λ+Iδ≥0；在所选 32 个自由节点求解原始场变化，保持非负节点及边界迹。间隙从 {fmt(u[0]['gap_before'])} 变为 {fmt(u[-1]['gap_after'])}。",
        '', '每步核对 BR=-I、真实能量变化与二次展开、接触电荷响应、非负性、平衡残差以及条件性间隙不增。原始初值是在已求解场上叠加非负扰动；选择单元的策略为本算例的残量启发式。这验证所执行更新的公式与验收条件，不构成任意选点策略或一般器件上的全局收敛定理。',
        '', '![局部更新](../results/figures/m4_updates.png)',
        '', '## 8. M5：给定有限区间上的 PB 对照','',
        table(d['m5_poisson_boltzmann'],['theta','Q_poisson_boltzmann','Q_depletion','relative_difference_percent','right_derivative','first_integral_residual']),
        '', '![PB](../results/figures/m5_poisson_boltzmann.png)',
        '', '比较固定 b=0.5、ℓ=2，边界 u(0)=b、u(ℓ)=0 的有限区间 PB 问题与耗尽模型。检查边值求解残差，并用独立第一积分 Q²−u′(ℓ)²=2[b+θ(exp(−b/θ)−1)] 核验电荷。右端导数单独报告，不能把有限区间 Dirichlet 边界当成精确中性零场边界。所测 θ 减小时差异降低，但这不确定一般器件的模型适用边界。',
        '', *priority_report(table,fmt), '## 10. 完整检查与复现','',
        '正文取舍及新增实验的结论范围见第 9.4 节。尚未做实验器件对照、输运模型对照、噪声稳健性或不同独立离散的交叉验证，不据此宣称这些能力。',
        '', '复现：在 experiments 目录执行 `python run_all.py`。脚本先运行回归测试，再在临时目录完整生成结果；验证成功后替换发布目录，失败则保留旧结果并返回非零退出状态。依赖版本和源文件哈希见运行清单。',
        '', '- [运行清单](../results/tables/run_manifest.json)',
        '- [回归测试日志](../results/tables/regression_tests.log)',
        '- [结果验证 JSON](../results/tables/verification.json)',
        '- [结果验证日志](../results/tables/verification.log)',
        '- [512×256 参考场](../results/fields/reference_512x256.npz)',
        '', '<details><summary>展开全部数值检查</summary>','',table(ver['checks'],['check','passed','detail']),
        '', '</details>','', '所有图同时提供 PNG 和矢量 PDF；原始数据保存在 results/tables。']
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'wrote report: {REPORT}',flush=True)


if __name__=='__main__':build_report()
