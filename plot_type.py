

# 5. 元論文の Fig. 1 に近い体裁でプロット
from matplotlib.ticker import (LogLocator,FuncFormatter,NullFormatter)

# -5/3 の基準線

n_x_line = np.logspace(np.log10(5.0e-7),np.log10(3.0e-1),300)

# 基準線の上下位置
# 大きくすると上、小さくすると下に移動する
reference_coefficient = 0.65
n_y_line = (reference_coefficient * n_x_line**(-5.0 / 3.0))

# 図の作成

fig, ax = plt.subplots(figsize=(7.2, 7.0),dpi=130)
ax.set_xscale("log")
ax.set_yscale("log")

# 計算結果

ax.plot(n_x_normalized,n_y_normalized,linestyle="none",marker="x",color="black",markersize=5.5,markeredgewidth=1.1)

# 傾き -5/3 の直線

ax.plot(n_x_line,n_y_line,color="black",linewidth=1.6)

# 軸範囲

ax.set_xlim(1.0e-8,1.0e2)
ax.set_ylim(1.0e-10,1.0e14)

# 横軸の設定　# 目盛りは10倍ごと　# 数字は　# 10^-8, 10^-6, 10^-4, 10^-2, 10^0, 10^2　# だけ表示

ax.xaxis.set_major_locator(LogLocator(base=10.0,subs=(1.0,),numticks=100))

def x_formatter(value, position):
    """
    横軸では偶数指数だけ数字を表示する。
    """
    if value <= 0:
        return ""

    exponent = int(np.round(np.log10(value)))

    if np.isclose(value, 10.0**exponent):
        if -8 <= exponent <= 2 and exponent % 2 == 0:
            return rf"$10^{{{exponent}}}$"

    return ""

ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))

# 各10倍区間の中に 2,3,...,9 の小目盛りを入れる
ax.xaxis.set_minor_locator(LogLocator(base=10.0,subs=np.arange(2, 10) * 0.1,numticks=200))
ax.xaxis.set_minor_formatter(NullFormatter())


# 縦軸の設定　# 目盛りは　# 10^-10, 10^-9, ..., 10^14　# と10倍ごと　# 数字は元論文と同様に　# 10^-10, 10^0, 10^10　# のみ表示

ax.yaxis.set_major_locator(LogLocator(base=10.0,subs=(1.0,),numticks=100))

def y_formatter(value, position):
    """
    縦軸では -10, 0, 10 の指数だけ数字を表示する。
    ただし、目盛り線そのものは10倍ごとに存在する。
    """
    if value <= 0:
        return ""

    exponent = int(np.round(np.log10(value)))

    if np.isclose(value, 10.0**exponent):
        if exponent in (-10, 0, 10):
            return rf"$10^{{{exponent}}}$"

    return ""

ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))

# 各10倍区間の中の小目盛り
ax.yaxis.set_minor_locator(LogLocator(base=10.0,subs=np.arange(2, 10) * 0.1,numticks=500))
ax.yaxis.set_minor_formatter(NullFormatter())

# 軸ラベル

ax.set_xlabel(r"$k/k_d$",fontsize=17,labelpad=8)
ax.set_ylabel(r"$E_e(k/k_d)$",fontsize=17,labelpad=14)

# 元論文にはグラフ上部のタイトルと凡例がない

# タイトルを付けない
ax.set_title("")

# 凡例を付けない　# 目盛りを四方向すべて内向きにする

ax.tick_params(axis="both",which="major",direction="in",top=True,right=True,length=10,width=1.2,labelsize=12,pad=5)
ax.tick_params(axis="both",which="minor",direction="in",top=True,right=True,length=4,width=0.8)

# 外枠

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("black")
    spine.set_linewidth(1.3)

# 元論文にはグリッド線がない
ax.grid(False)

# 余白調整
fig.subplots_adjust(left=0.17,right=0.97,bottom=0.14,top=0.97)

plt.show()
