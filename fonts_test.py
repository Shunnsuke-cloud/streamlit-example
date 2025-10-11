import matplotlib.font_manager as fm

fonts = set([f.name for f in fm.fontManager.ttflist])
for font in sorted(fonts):
    print(font)
