#!/usr/bin/env python3
import sys
import pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn


dfTSNE = pandas.read_csv(sys.argv[1])
label_col, label_style = 'GeoGroup', 'BerryColour'
n_colors = len(set(dfTSNE[label_col].values))
g = seaborn.relplot(data=dfTSNE, x='tsne1',y='tsne2', hue=label_col,
                palette=seaborn.color_palette(palette="tab20",n_colors=n_colors),
                style=label_style, picker = True
                )


def onpick(event):
    # Получаем координаты клика
    x_click = event.mouseevent.xdata
    y_click = event.mouseevent.ydata
    
    radius = 1.0
    distances = np.sqrt((dfTSNE['tsne1'] - x_click)**2 + (dfTSNE['tsne2'] - y_click)**2)
    indices = np.where(distances <= radius)[0]
    
    if len(indices) == 0:
        return
    
    print(f"\n{'='*50}\n")
    for i, idx in enumerate(indices):
        print(f"ID: {dfTSNE.loc[idx, 'ID']}")
        print(f"  tsne1: {dfTSNE.loc[idx, 'tsne1']:.3f}")
        print(f"  tsne2: {dfTSNE.loc[idx, 'tsne2']:.3f}")
        print(f"  GeoGroup: {dfTSNE.loc[idx, 'GeoGroup']}")
        print(f"  BerryColour: {dfTSNE.loc[idx, 'BerryColour']}")

g.figure.canvas.mpl_connect("pick_event", onpick)

plt.show()
