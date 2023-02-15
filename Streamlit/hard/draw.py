# 导入库
from pyecharts.charts import Timeline, Line
import pyecharts.options as opts
from pyecharts.globals import ThemeType

# --------------------------分箱概率 - 固定分箱------------------------------------------------
def mark_time_box(config):
    time_box = Timeline(init_opts=opts.InitOpts(theme=ThemeType.WALDEN))
    fea_list=list(config["fea"].keys())

    for fea in fea_list:
        int_line = (
            Line()
            .add_xaxis([i for i in range(len(config["fea"][fea]["bins"])-1)])
            .add_yaxis("原概率",
                       config["fea"][fea]["rate"],
                       is_smooth=True,
                       linestyle_opts=opts.LineStyleOpts(width=0),
                       is_symbol_show=False,  # 是否显示数字
                       areastyle_opts=opts.AreaStyleOpts(opacity=0.3,), #透明度
                       itemstyle_opts=opts.ItemStyleOpts(color='blue')
                       )
            .add_yaxis("拟合概率",
                       config["fea"][fea]["fit_rate"],
                       is_smooth=True,
                       linestyle_opts=opts.LineStyleOpts(width=0),
                       is_symbol_show=False,  # 是否显示数字
                       areastyle_opts=opts.AreaStyleOpts(opacity=0.3,), #透明度
                       itemstyle_opts=opts.ItemStyleOpts(color='red')
                       )

            .set_global_opts(
                title_opts=opts.TitleOpts(title=fea,
                                          pos_left='center',
                                          title_textstyle_opts=opts.TextStyleOpts(
                                              font_size=30, font_weight='bolder', color='#516b91')
                                          ),
                legend_opts=opts.LegendOpts(pos_top='7%',),
                # legend_opts=opts.LegendOpts(is_show=False,),
                # yaxis_opts=opts.AxisOpts(is_show=False,),
                xaxis_opts=opts.AxisOpts(
                    axistick_opts=opts.AxisTickOpts(is_align_with_label=True),
                    #                 is_scale=False,
                    # is_show=False,
                    boundary_gap=False,
                ),
            )
        )

        time_box.add(int_line, time_point=fea)
    # --------------------------加入时间线------------------------------------------------
    time_box.add_schema(
        is_auto_play=False,  # 自启动
        play_interval=1000,
    )
    return time_box
