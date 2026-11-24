"""Report observations without treating expected convergence as acceptance."""
from .paths import TABLES
import csv


def report(table,fmt):
    def read(name):
        with (TABLES/(name+'.csv')).open() as f:return list(csv.DictReader(f))
    p1=read('p1_offgrid_convergence');s1=read('p1_offgrid_summary')
    p2=read('p2_bias_summary');finest=max(int(r['nx']) for r in p2)
    fine=[r for r in p2 if int(r['nx'])==finest]
    p3=read('p3_energy_charge_counterexample')
    ends=[]
    for case in ('fixed_multiplier_flux','varying_multiplier_recovery'):
        g=[r for r in p3 if r['case']==case]
        a,z=g[0],g[-1]
        ends.append(dict(case=case,D_start=a['D'],D_end=z['D'],gap_start=a['gap'],gap_end=z['gap'],
            gap_reduction_percent=100*(float(a['gap'])-float(z['gap']))/float(a['gap']),
            Q_flux_start=a['Q_flux'],Q_flux_end=z['Q_flux'],
            Q_recovered_start=a['Q_recovered'],Q_recovered_end=z['Q_recovered']))
    increases=sum(int(r['charge_error_increase_count']) for r in s1)
    changes=sum(int(r['charge_sign_changes']) for r in s1)
    widths=[float(r['relative_width_percent']) for r in fine]
    leveltext=[]
    for r in fine:
        prev=next(x for x in p2 if x['b']==r['b'] and int(x['nx'])==256)
        leveltext.append(f"b={r['b']}：{float(prev['relative_width_percent']):.2f}% → {float(r['relative_width_percent']):.2f}%")
    md=['## 9. 新增优先实验与客观分析','',
        '### 9.1 P1：固定偏压下的非对齐界面收敛','',
        '保持域长 ℓ=2，在 b=0.05、2/9、0.37、1.3 四个固定偏压下使用 n=16 至 1024 的七层二倍加密。解析界面 w=√(2b) 在全部 28 组网格上均不落在节点上。b=2/9 的单元内相位在 1/3 与 2/3 之间交替；其余三组展示非周期的相位变化。误差积分在解析界面处分段，不将网格穿越误差混入求积误差。',
        '',table(s1,['b','fit_order_energy_error','fit_order_gap','fit_order_Q_direct_abs_error','fit_order_Q_balance_abs_error','charge_error_increase_count','charge_sign_changes'],
            ['偏压','能量拟合阶','间隙拟合阶','直接电荷拟合阶','恢复电荷拟合阶','电荷绝对误差上升次数','电荷误差变号次数']),
        '', '拟合阶为最后四层 n=128、256、512、1024 的 log(error)-log(h) 最小二乘斜率；不是新收敛定理。逐层阶、带符号误差、相位和 |误差|/h² 全部保存在原始表中，首层逐层阶明确记为不适用。',
        '',table(s1,['b','scaled_charge_min','scaled_charge_max','finest_charge_error','effectivity_min','effectivity_max'],
            ['偏压','最小绝对电荷误差/h²','最大绝对电荷误差/h²','最细层电荷误差','最小间隙效率比','最大间隙效率比']),
        '',f'四组序列合计出现 {increases} 次恢复电荷绝对误差上升、{changes} 次带符号误差变号。相位变化及误差抵消会使相邻层观察阶剧烈波动；因此不能用某一对网格的小误差或高观察阶宣传超收敛，也不能声称每次加密都改善电荷。能量界、梯度误差界及适用时的界面区间仍逐例核验。',
        '', '![非对齐界面收敛与相位](../results/figures/p1_offgrid_convergence.png)',
        '', '### 9.2 P2：多个二维偏压下的电荷数值区间','',
        '保持 Ω=[0,2]×[0,1]、左侧居中接触 y∈[0.25,0.75]、右侧零电势及其余绝缘边界不变。在 b=0.05、0.2、0.5、1.5、3 下，分别采用 128×64、256×128、512×256 网格；割线步长取偏压的 5%、10%、20%。共计算 105 组原始/对偶场，构造 45 个电荷区间。每一个中心及偏移偏压样本均检查 KKT、非负性、平衡及独立求积。',
        '', '下表选取各偏压最细网格上所测三个步长中的最窄区间；该选择只依赖区间宽度，不用参考电荷调参。原始数据保留所有步长，并另列三个有效区间的交集。相对宽度的分母为最细网格恢复值；该值是近似，不是真值。',
        '',table(fine,['b','Q_reference','best_step','Q_lower','Q_upper','relative_width_percent','reference_mesh_change_percent','active_free_fraction'],
            ['偏压','最细恢复值','所选步长','区间下端','区间上端','全宽/参考值 %','256→512 恢复值变化 %','零障碍活跃节点比例']),
        '',f'最细层相对全宽的范围为 {min(widths):.2f}%–{max(widths):.2f}%。256→512 层的相对全宽变化：'+'；'.join(leveltext)+'。',
        '', '这些数据区分了点估计的网格稳定性与能量区间的宽度。活跃比例仅指该离散求解中的自由节点状态，不能借用一维 b=2 的转折阈值判断二维耗尽分支。所测三个相对步长不代表连续最优步长；宽区间可能同时反映能量界较松和割线偏差，不能把原因全部归为某一项。',
        '', '![二维多偏压电荷区间](../results/figures/p2_bias_charge_bounds.png)',
        '', '未使用向外舍入，以上仍是已检查浮点可行性的数值区间。所有细网格点值落入相应区间仅是相容性检查，不构成独立真值验证，也不提供二维点态电容界。',
        '', '### 9.3 P3：能量改善而电荷误差增大的受控算例','',
        '取一维 b=1/2、ℓ=2，解析电荷 Q*=1、能量 E*=1/3、界面 x=1。保持同一个 n=64 原始可行 P1 场，改变非负分片常数乘子与平衡分片线性通量。由于原始能量固定，对偶能量上升必然缩小间隙。所有构造都核验非负性、边界迹、σ′=1−λ、独立求积及间隙分解。',
        '', '**A. 固定乘子的通量校正。** 在 x<1 取 λ=0，在 x≥1 取 λ=1.05。对应恢复电荷固定为 0.9875。起始通量通过常数偏移使通量电荷恰为 1，再沿调和方向逐渐校正到当前乘子的最优通量。对偶能量逐步上升，通量电荷误差从 0 增至 1.25%；恢复电荷始终不变。这展示固定乘子校正只能达到该乘子确定的恢复值。',
        '', '**B. 改变乘子的最优平衡通量。** 在 x<1 取 λ=0；在 [1,2] 取 λ(t)=1+0.2(1−t)f(x)+0.05t，其中 f 在 [1,1.5) 为 1、在 [1.5,2] 为 −3，t 从 0 增至 0.8。因为 ∫ηf=0，初始恢复电荷恰为真值；每一步都使用该乘子的最优通量，所以通量电荷与恢复电荷相同。',
        '', '该族可直接计算 Q(t)=1−t/80，以及 E*−D(t)=(1−t)²/300−t(1−t)/1200+t²/3840。后者在 [0,0.8] 上严格下降，电荷绝对误差严格上升；数值表与这些闭式表达逐项比较。该路径用于隔离逻辑关系，不是实际局部 QP 选点策略的运行轨迹。',
        '',table(ends,['case','D_start','D_end','gap_reduction_percent','Q_flux_start','Q_flux_end','Q_recovered_start','Q_recovered_end']),
        '', '![能量与电荷误差受控对照](../results/figures/p3_energy_charge_counterexample.png)',
        '', '案例 B 将恢复电荷误差从 0 增至 1%，同时明显缩小能量间隙。起点的零电荷误差来自加权抵消，不代表屏蔽分布准确。此反例说明能量改善不推出具体电荷误差逐步改善；它不否定有效的能量误差界，也不说明整个迭代过程最终不能收敛。',
        '', '### 9.4 对论文结论与正文取舍的更新','',
        '- 非对齐实验扩展了 M1 的证据范围：可报告能量与间隙的观察阶，并明确恢复电荷存在相位相关波动；不得把对齐例的逐层二阶趋势写成无条件结论。',
        '- 多偏压区间使二维证据不再局限于 b=0.5。正文宜同时列参考值、网格间变化和区间相对全宽，避免用稳定点值代替误差保证。当前数据仍未验证一般几何或真实器件精度。',
        '- 受控反例直接支撑第 4 节关于固定屏蔽偏差与输出非单调性的限定，建议在正文放一个紧凑图或首末状态表，并将完整构造放附录。',
        '- 正文用新的非对齐图合并原 M1，使用多偏压短表扩展 M3，保留 S1 几何分解与一个实际 M4 更新结果。其余参数表、固定乘子反例细节及日志放补充材料；不把新增图全部堆入正文。',
        '- 实验补强了现有数学结构的验证；尚未提供局部更新效率优势、一般网格超收敛、二维点态电容认证或真实器件准确性的证据。', '']
    return md
