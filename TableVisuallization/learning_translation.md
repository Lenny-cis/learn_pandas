学习内容
# 分享样式
使用`df.style.export`导出样式，使用`df.style.use`来导入样式。即使样式是根据数据制定的，也可以共享，样式会根据数据重新评估。

# 局限
* 只能对DataFrame作用
* 行或列索引不一定要唯一，但是一些样式要求索引唯一
* 没有多文本，性能不佳
* 只能应用样式，不能插入新的HTML

# 其他有用的东西
## 窗体小部件
`from ipywidgets import widgets`
## 放大
？？？
## 冻结表头
使用`.set_sticky`方法来冻结表头，对分层索引依然有效
## HTML转义
如果要再HTML里展示HTML，展示器可能无法显示，可以使用`escape`格式选项来处理。

# 导出到Excel
支持使用`OpenPyXL`和`XlsxWriter`引擎可以把样式导出到工作页
* 包括：`background-color`, `border-style`, `border-width`, `border-color`, `color`, `font-family`, `font-style`, `font-weight`, `text-align`, `text-decoration`, `vertical-align`, `white-space: nowrap`
* 使用`border`简写会覆盖`border-style`等的属性。
* 仅支持CSS2命名的颜色和hex格式的颜色`#rgb`或`#rrggbb`
* 以下伪CCS属性也可以指定为excel属性：
> * `number-format`
> * `border-style`Excel特定样式：'hair', 'mediumDashDot', 'dashDotDot', 'mediumDashDotDot', 'dashDot', 'slantDashDot', 'mediumDashed'
表级样式和数据单元格样式不能被导出到excel，单元格必须要被`Styler.apply`或者`Styler.applymap`方法应用
# 导出到LaTeX
