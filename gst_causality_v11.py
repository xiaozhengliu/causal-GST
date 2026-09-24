#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gst_causality.py - 全局信号因果映射分析工具箱

功能：
- 计算体素级时间序列与全局信号之间的双向因果强度
- 支持三种方法：梁信息流、格兰杰因果、转移熵
- 支持NIfTI格式输入/输出
- 提供因果密度和因果平衡指标
- 支持 leave-one-out 全局信号（消除自泄漏）

版本：1.1
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 可选依赖（若需NIfTI支持）
try:
    import nibabel as nib
except ImportError:
    nib = None
    print("警告：未安装 nibabel，NIfTI 功能不可用")


# =============================================================================
# 1. 梁信息流 (Liang Information Flow) - 修正版本
# =============================================================================
class LiangInformationFlow:
    """
    梁信息流双变量因果分析 (Liang, 2014, Phys. Rev. E)
    修正了方向计算中的协方差使用错误，确保两个方向不对称。
    """

    @staticmethod
    def bivariate_flow(x, y, dt=1):
        """
        计算两个时间序列之间的梁信息流

        参数
        ----------
        x, y : 1D array, shape (n_timepoints,)
        dt : 采样间隔（默认1）

        返回
        -------
        T_y_to_x : float
            从 y 到 x 的信息流（正值表示 y 驱动 x）
        T_x_to_y : float
            从 x 到 y 的信息流（正值表示 x 驱动 y）
        """
        n = len(x)
        # 计算时间导数（中心差分）
        dx_dt = np.zeros(n)
        dy_dt = np.zeros(n)
        dx_dt[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        dy_dt[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
        # 边界使用单侧差分
        dx_dt[0] = (x[1] - x[0]) / dt
        dx_dt[-1] = (x[-1] - x[-2]) / dt
        dy_dt[0] = (y[1] - y[0]) / dt
        dy_dt[-1] = (y[-1] - y[-2]) / dt

        # 方差与协方差（统一使用样本统计量 ddof=1）
        C11 = np.var(x, ddof=1)
        C22 = np.var(y, ddof=1)
        C12 = np.cov(x, y, ddof=1)[0, 1]

        # 正确的协方差组合
        C_x_dx = np.cov(x, dx_dt, ddof=1)[0, 1]   # Cov(x, dx/dt)
        C_y_dx = np.cov(y, dx_dt, ddof=1)[0, 1]   # Cov(y, dx/dt)
        C_y_dy = np.cov(y, dy_dt, ddof=1)[0, 1]   # Cov(y, dy/dt)
        C_x_dy = np.cov(x, dy_dt, ddof=1)[0, 1]   # Cov(x, dy/dt)

        # T_{y->x} = (C11*C12*C_{y,dx} - C12^2*C_{x,dx}) / (C11^2*C22 - C11*C12^2)
        denom1 = C11**2 * C22 - C11 * C12**2
        if abs(denom1) > 1e-12:
            T_y_to_x = (C11 * C12 * C_y_dx - C12**2 * C_x_dx) / denom1
        else:
            T_y_to_x = 0.0

        # T_{x->y} = (C22*C12*C_{x,dy} - C12^2*C_{y,dy}) / (C22^2*C11 - C22*C12^2)
        denom2 = C22**2 * C11 - C22 * C12**2
        if abs(denom2) > 1e-12:
            T_x_to_y = (C22 * C12 * C_x_dy - C12**2 * C_y_dy) / denom2
        else:
            T_x_to_y = 0.0

        return T_y_to_x, T_x_to_y

    @staticmethod
    def normalized_flow(x, y, dt=1):
        """返回归一化的信息流强度（绝对值在0~1之间）"""
        T_yx, T_xy = LiangInformationFlow.bivariate_flow(x, y, dt)
        # 使用总变化率作为归一化因子
        H_rate = np.mean(np.abs(np.diff(x))) + np.mean(np.abs(np.diff(y)))
        if H_rate < 1e-12:
            return 0.0, 0.0
        return abs(T_yx) / (abs(T_yx) + H_rate), abs(T_xy) / (abs(T_xy) + H_rate)


# =============================================================================
# 2. 格兰杰因果 (Granger Causality)
# =============================================================================
class GrangerCausality:
    """基于线性回归的格兰杰因果分析"""

    @staticmethod
    def bivariate_gc(x, y, max_lag=1):
        """
        双变量格兰杰因果检验

        参数
        ----------
        x : 1D array, 被预测的目标时间序列
        y : 1D array, 候选的预测源时间序列
        max_lag : int, 滞后阶数

        返回
        -------
        F_stat : float
        p_value : float
        gc_score : float  因果强度 (log(F+1))
        """
        n = len(x)
        # 构建滞后矩阵
        X_lagged = []
        for lag in range(1, max_lag + 1):
            X_lagged.append(x[max_lag - lag:n - lag])
        X_lagged = np.array(X_lagged).T

        Y_lagged = []
        for lag in range(1, max_lag + 1):
            Y_lagged.append(y[max_lag - lag:n - lag])
        Y_lagged = np.array(Y_lagged).T

        target = x[max_lag:]

        # 受限模型（仅X自回归）
        model_r = LinearRegression()
        model_r.fit(X_lagged, target)
        rss_r = np.sum((target - model_r.predict(X_lagged)) ** 2)

        # 非受限模型（X+Y）
        X_full = np.hstack([X_lagged, Y_lagged])
        model_f = LinearRegression()
        model_f.fit(X_full, target)
        rss_f = np.sum((target - model_f.predict(X_full)) ** 2)

        df1 = max_lag
        df2 = n - 2 * max_lag - 1
        if rss_f == 0 or df2 <= 0:
            return 0.0, 1.0, 0.0

        # 防止数值误差导致 rss_r < rss_f
        if rss_r < rss_f:
            rss_r = rss_f

        F_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
        p_value = 1 - stats.f.cdf(F_stat, df1, df2)
        gc_score = np.log(F_stat + 1) if F_stat > 0 else 0.0
        return F_stat, p_value, gc_score


# =============================================================================
# 3. 转移熵 (Transfer Entropy)
# =============================================================================
class TransferEntropy:
    """基于离散化估计的转移熵（非线性）"""

    @staticmethod
    def bivariate_te(x, y, lag=1, n_bins=10):
        """计算从 y 到 x 的转移熵"""
        n = len(x)

        # 修复：去除双重 np.percentile 嵌套
        bins_x = np.percentile(x, np.linspace(0, 100, n_bins + 1)[1:-1])
        bins_y = np.percentile(y, np.linspace(0, 100, n_bins + 1)[1:-1])

        # 处理重复边界（对常数区域做微小抖动）
        if len(np.unique(bins_x)) < len(bins_x):
            x_jittered = x + np.random.normal(0, 1e-9, size=n)
            bins_x = np.percentile(x_jittered, np.linspace(0, 100, n_bins + 1)[1:-1])
        if len(np.unique(bins_y)) < len(bins_y):
            y_jittered = y + np.random.normal(0, 1e-9, size=n)
            bins_y = np.percentile(y_jittered, np.linspace(0, 100, n_bins + 1)[1:-1])

        x_disc = np.digitize(x, bins_x)
        y_disc = np.digitize(y, bins_y)

        x_t = x_disc[lag:]
        x_t_lag = x_disc[:n - lag]
        y_t_lag = y_disc[:n - lag]

        def joint_entropy(vectors):
            """计算联合熵"""
            arr = np.column_stack(vectors)
            _, counts = np.unique(arr, axis=0, return_counts=True)
            probs = counts / len(arr)
            return -np.sum(probs * np.log(probs + 1e-12))

        # H(x_t | x_{t-lag})
        H_xt_xtlag = joint_entropy([x_t, x_t_lag]) - joint_entropy([x_t_lag])
        # H(x_t | x_{t-lag}, y_{t-lag})
        H_xt_xtlag_ytlag = joint_entropy([x_t, x_t_lag, y_t_lag]) - joint_entropy([x_t_lag, y_t_lag])

        te = H_xt_xtlag - H_xt_xtlag_ytlag
        return max(0.0, te)


# =============================================================================
# 4. 全局信号因果映射主类
# =============================================================================
class GlobalSignalCausalityMapper:
    """
    计算全脑每个体素与全局信号之间的双向因果强度

    参数
    ----------
    brain_data : ndarray, shape (n_voxels, n_timepoints)
        功能数据（可以是全脑扁平或仅灰质，由mask决定）
    gm_mask : ndarray, shape (n_voxels,)
        灰质掩码（1/True为灰质）；若为None则使用所有体素
    method : str, {'liang', 'granger', 'transfer_entropy'}
        因果分析方法
    max_lag : int
        滞后阶数（Granger和转移熵使用）
    loo_global : bool
        是否使用 leave-one-out 全局信号（消除自泄漏）。
        若为 True，每个体素的全局信号 = 除该体素外所有灰质体素的平均。
        计算量较大，但结果更严谨。默认 False 以兼容旧行为。
    """

    def __init__(self, brain_data, gm_mask=None, method='liang', max_lag=1, loo_global=False):
        self.brain_data = brain_data
        self.gm_mask = gm_mask
        self.method = method.lower()
        self.max_lag = max_lag
        self.loo_global = loo_global

        # 如果掩码是多维的，展平
        if gm_mask is not None:
            self.flat_mask = np.asarray(gm_mask).flatten()
            self.gm_indices = np.where(self.flat_mask)[0]
            self.n_voxels_total = len(self.flat_mask)
        else:
            self.flat_mask = None
            self.gm_indices = np.arange(brain_data.shape[0])
            self.n_voxels_total = brain_data.shape[0]

        # 提取灰质数据
        if gm_mask is not None:
            self.gm_data = brain_data[self.gm_indices, :]
        else:
            self.gm_data = brain_data

        # 标准化（每个体素Z-score）
        self.scaler = StandardScaler()
        self.gm_data_z = self.scaler.fit_transform(self.gm_data.T).T

        # 修复：处理零方差体素产生的 NaN
        self.gm_data_z = np.nan_to_num(self.gm_data_z, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算全局信号（灰质平均值）
        self.global_signal = np.mean(self.gm_data_z, axis=0)

    def _get_loo_global_signal(self, idx):
        
        n_gm = self.gm_data_z.shape[0]
    
        return (n_gm * self.global_signal - self.gm_data_z[idx]) / (n_gm - 1)

    def compute_causality_map(self):
        """
        计算每个灰质体素与全局信号的双向因果强度

        返回
        -------
        to_global : ndarray, shape (n_voxels_total,)
            体素→全局的因果强度（非灰质区域为0）
        from_global : ndarray, shape (n_voxels_total,)
            全局→体素的因果强度（非灰质区域为0）
        """
        n_gm = self.gm_data_z.shape[0]
        to_gm = np.zeros(n_gm)
        from_gm = np.zeros(n_gm)

        for i in range(n_gm):
            voxel_ts = self.gm_data_z[i]

            # 根据是否使用 leave-one-out 选择全局信号
            if self.loo_global:
                global_ts = self._get_loo_global_signal(i)
            else:
                global_ts = self.global_signal

            if self.method == 'liang':
                # bivariate_flow(x, y) 返回 (T_y_to_x, T_x_to_y)
                # x=voxel_ts, y=global_ts
                # T_y_to_x = global→voxel,  T_x_to_y = voxel→global
                T_gv, T_vg = LiangInformationFlow.bivariate_flow(voxel_ts, global_ts)
                to_gm[i]   = T_vg   # 体素→全局
                from_gm[i] = T_gv   # 全局→体素

            elif self.method == 'granger':
                # 体素→全局：体素预测全局
                _, _, gc_vg = GrangerCausality.bivariate_gc(global_ts, voxel_ts, self.max_lag)
                # 全局→体素：全局预测体素
                _, _, gc_gv = GrangerCausality.bivariate_gc(voxel_ts, global_ts, self.max_lag)
                to_gm[i] = gc_vg
                from_gm[i] = gc_gv

            elif self.method == 'transfer_entropy':
                te_vg = TransferEntropy.bivariate_te(global_ts, voxel_ts, self.max_lag)
                te_gv = TransferEntropy.bivariate_te(voxel_ts, global_ts, self.max_lag)
                to_gm[i] = te_vg
                from_gm[i] = te_gv

            else:
                raise ValueError(f"未知方法: {self.method}")

        # 映射到全脑（非灰质置零）
        full_to = np.zeros(self.n_voxels_total, dtype=np.float32)
        full_from = np.zeros(self.n_voxels_total, dtype=np.float32)
        if self.gm_mask is not None:
            full_to[self.gm_indices] = to_gm
            full_from[self.gm_indices] = from_gm
        else:
            full_to = to_gm
            full_from = from_gm

        return full_to, full_from

    def compute_causality_density(self):
        """因果密度 = |体素→全局| + |全局→体素|"""
        to_g, from_g = self.compute_causality_map()
        return np.abs(to_g) + np.abs(from_g)

    def compute_causal_balance(self):
        """因果平衡 = 体素→全局 - 全局→体素（正值表示该体素驱动全局）"""
        to_g, from_g = self.compute_causality_map()
        return to_g - from_g


# =============================================================================
# 5. NIfTI 辅助函数
# =============================================================================
def load_and_compute_causality(nifti_file, mask_file, method='liang', max_lag=1,
                               output_prefix=None, loo_global=False):
    """
    直接从NIfTI文件计算因果图并保存

    参数
    ----------
    nifti_file : str
        4D功能图像路径
    mask_file : str
        灰质掩码路径（3D，二值）
    method : str
        因果方法
    max_lag : int
        滞后阶数
    output_prefix : str, optional
        输出文件前缀（不含扩展名），若为None则不保存
    loo_global : bool
        是否使用 leave-one-out 全局信号

    返回
    -------
    results : dict
        包含 'to_global', 'from_global', 'balance', 'density' 的3D数组
    """
    if nib is None:
        raise ImportError("需要安装 nibabel")

    # 加载数据
    img = nib.load(nifti_file)
    data = img.get_fdata()
    mask_img = nib.load(mask_file)
    mask = mask_img.get_fdata().astype(bool)
    affine = mask_img.affine
    header = mask_img.header

    # 重塑为 (体素总数, 时间)
    orig_shape = data.shape[:-1]
    n_time = data.shape[-1]
    data_flat = data.reshape(-1, n_time)
    mask_flat = mask.flatten()

    # 计算因果
    mapper = GlobalSignalCausalityMapper(data_flat, mask_flat, method=method, 
                                         max_lag=max_lag, loo_global=loo_global)
    to_global, from_global = mapper.compute_causality_map()
    balance = mapper.compute_causal_balance()
    density = mapper.compute_causality_density()

    # 重塑回3D
    to_3d = to_global.reshape(orig_shape)
    from_3d = from_global.reshape(orig_shape)
    balance_3d = balance.reshape(orig_shape)
    density_3d = density.reshape(orig_shape)

    # 保存
    if output_prefix is not None:
        nib.save(nib.Nifti1Image(to_3d, affine, header),
                 f"{output_prefix}_to_global.nii.gz")
        nib.save(nib.Nifti1Image(from_3d, affine, header),
                 f"{output_prefix}_from_global.nii.gz")
        nib.save(nib.Nifti1Image(balance_3d, affine, header),
                 f"{output_prefix}_balance.nii.gz")
        nib.save(nib.Nifti1Image(density_3d, affine, header),
                 f"{output_prefix}_density.nii.gz")

    return {
        'to_global': to_3d,
        'from_global': from_3d,
        'balance': balance_3d,
        'density': density_3d
    }


# =============================================================================
# 6. 命令行示例 / 测试
# =============================================================================
if __name__ == "__main__":
    # 生成模拟数据测试
    np.random.seed(42)
    n_voxels = 100
    n_time = 150

    # 模拟：前20个体素受全局信号较强驱动（全局→体素方向强）
    # 后20个体素对全局信号有较强驱动（体素→全局方向强）
    global_signal = np.sin(np.linspace(0, 4*np.pi, n_time))
    global_signal = (global_signal - np.mean(global_signal)) / np.std(global_signal)

    brain = np.zeros((n_voxels, n_time))
    for i in range(n_voxels):
        if i < 20:
            # 前20个体素：主要由全局信号驱动
            brain[i] = 0.8 * global_signal + 0.3 * np.random.randn(n_time)
        elif i >= 80:
            # 后20个体素：对全局信号有贡献（全局信号包含这些体素的成分）
            # 这里我们直接让后20个体素与全局信号高度相关且相位领先
            brain[i] = 0.8 * np.roll(global_signal, -2) + 0.3 * np.random.randn(n_time)
        else:
            brain[i] = 0.4 * global_signal + 0.6 * np.random.randn(n_time)

    mask = np.ones(n_voxels, dtype=bool)  # 全部视为灰质

    # 使用梁信息流（默认非 leave-one-out，以兼容旧行为）
    mapper = GlobalSignalCausalityMapper(brain, mask, method='liang')
    to_g, from_g = mapper.compute_causality_map()
    balance = mapper.compute_causal_balance()

    print("体素→全局 均值:", np.mean(to_g))
    print("全局→体素 均值:", np.mean(from_g))
    print("因果平衡 (前20体素均值):", np.mean(balance[:20]))
    print("因果平衡 (中间60体素均值):", np.mean(balance[20:80]))
    print("因果平衡 (后20体素均值):", np.mean(balance[80:]))
    print("（前20体素应接近0或略负（被驱动），后20体素应为正（驱动全局））")
